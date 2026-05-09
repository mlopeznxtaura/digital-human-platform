"""
FastAPI + Socket.io NPC interaction API.
REST for session management, WebSocket for real-time NPC conversation.
SDKs: FastAPI, python-socketio, Redis, LangGraph
"""
import os
import json
import time
import uuid
from typing import Optional, Dict, Any

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

from npc.brain import NPCBrain, CHARACTERS, MAYA
from npc.memory import NPCMemoryStore

# App setup
app = FastAPI(title="Digital Human Platform API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Redis for session state
try:
    r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
    r.ping()
    REDIS_OK = True
except Exception:
    REDIS_OK = False
    _session_store: Dict[str, Any] = {}

# NPC instances (one per character)
_npcs: Dict[str, NPCBrain] = {}
_memories: Dict[str, NPCMemoryStore] = {}


def get_npc(character_name: str) -> NPCBrain:
    if character_name not in _npcs:
        persona = CHARACTERS.get(character_name, MAYA)
        memory = NPCMemoryStore(character_name=character_name)
        _memories[character_name] = memory
        _npcs[character_name] = NPCBrain(
            character=persona,
            model=os.environ.get("OLLAMA_MODEL", "mistral"),
            ollama_base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            memory_store=memory,
        )
    return _npcs[character_name]


def save_session(session_id: str, data: Dict):
    if REDIS_OK:
        r.setex(f"session:{session_id}", 3600, json.dumps(data))
    else:
        _session_store[session_id] = data


def get_session(session_id: str) -> Dict:
    if REDIS_OK:
        val = r.get(f"session:{session_id}")
        return json.loads(val) if val else {}
    return _session_store.get(session_id, {})


# --- REST endpoints ---

class ChatRequest(BaseModel):
    message: str
    character: str = "maya"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    character: str
    emotion: str
    session_id: str
    response_time_ms: float


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to an NPC character and get a response."""
    session_id = req.session_id or str(uuid.uuid4())
    if req.character not in CHARACTERS and req.character != "maya":
        raise HTTPException(status_code=404, detail=f"Character '{req.character}' not found")

    npc = get_npc(req.character)
    t0 = time.perf_counter()
    result = npc.respond(req.message, session_id=session_id)
    elapsed = (time.perf_counter() - t0) * 1000

    session = get_session(session_id)
    session.setdefault("messages", [])
    session["messages"].append({"role": "user", "content": req.message})
    session["messages"].append({"role": req.character, "content": result["response"]})
    session["character"] = req.character
    save_session(session_id, session)

    return ChatResponse(
        response=result["response"],
        character=result["character"],
        emotion=result["emotion"],
        session_id=session_id,
        response_time_ms=round(elapsed, 1),
    )


@app.get("/characters")
async def list_characters():
    return {
        name: {
            "name": persona.name,
            "backstory": persona.backstory[:100] + "...",
            "traits": persona.personality_traits,
        }
        for name, persona in CHARACTERS.items()
    }


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    return get_session(session_id)


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    if REDIS_OK:
        r.delete(f"session:{session_id}")
    else:
        _session_store.pop(session_id, None)
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": REDIS_OK,
        "loaded_npcs": list(_npcs.keys()),
        "characters_available": list(CHARACTERS.keys()),
    }


# --- Socket.io for streaming responses ---

@sio.event
async def connect(sid, environ, auth=None):
    session_id = auth.get("session_id", str(uuid.uuid4())) if auth else str(uuid.uuid4())
    await sio.save_session(sid, {"session_id": session_id, "character": "maya"})
    await sio.emit("connected", {"session_id": session_id}, to=sid)

@sio.event
async def disconnect(sid):
    pass

@sio.event
async def chat_stream(sid, data):
    """Stream NPC response token by token."""
    session = await sio.get_session(sid)
    session_id = session.get("session_id", sid)
    character_name = data.get("character", session.get("character", "maya"))
    message = data.get("message", "")

    npc = get_npc(character_name)
    await sio.emit("response_start", {"character": character_name, "emotion": "thinking"}, to=sid)

    full_response = ""
    try:
        for token in npc.stream_response(message, session_id=session_id):
            full_response += token
            await sio.emit("response_token", {"token": token}, to=sid)
    except Exception as e:
        await sio.emit("response_error", {"error": str(e)}, to=sid)

    await sio.emit("response_end", {
        "full_response": full_response,
        "character": character_name,
    }, to=sid)
