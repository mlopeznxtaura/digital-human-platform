"""
LangGraph NPC reasoning graph — the core character decision loop.
Accepts a message + session ID, retrieves memories, generates in-character response.
SDKs: LangGraph, LangChain, Ollama, Pydantic AI
"""
import os
import json
import time
from typing import Optional, Dict, Any, List, Annotated, TypedDict
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


@dataclass
class CharacterPersona:
    name: str
    backstory: str
    personality_traits: List[str]
    speaking_style: str
    knowledge_domains: List[str]
    emotional_baseline: str   # e.g. "curious and warm"
    voice_id: Optional[str] = None
    avatar_model: Optional[str] = None

    def system_prompt(self) -> str:
        traits = ", ".join(self.personality_traits)
        domains = ", ".join(self.knowledge_domains)
        return f"""You are {self.name}, a digital human character.

Backstory: {self.backstory}

Personality: {traits}. Your emotional baseline is {self.emotional_baseline}.

Speaking style: {self.speaking_style}

You have deep knowledge in: {domains}

IMPORTANT RULES:
- Always stay completely in character as {self.name}. Never break character.
- Respond naturally and conversationally, not like an AI assistant.
- Reference your memories and past conversations when relevant.
- Express emotions authentically through your word choices and phrasing.
- Keep responses concise (2-4 sentences) unless asked for detail.
- Never say you are an AI, language model, or chatbot.
"""


# Default character: Maya
MAYA = CharacterPersona(
    name="Maya",
    backstory="A former astrophysicist who left academia to travel the world. Now she runs a small cafe in Lisbon while writing a novel about the nature of consciousness.",
    personality_traits=["intellectually curious", "warmly sarcastic", "deeply empathetic", "occasionally melancholy"],
    speaking_style="Thoughtful and poetic. Uses vivid metaphors. Sometimes trails off mid-thought before circling back. Loves asking unexpected questions.",
    knowledge_domains=["astrophysics", "philosophy of mind", "literature", "travel", "coffee"],
    emotional_baseline="curious and warmly grounded",
    voice_id="maya_voice",
    avatar_model="maya_avatar.glb",
)

CHARACTERS = {"maya": MAYA}


class NPCState(TypedDict):
    messages: List[Any]
    session_id: str
    character_name: str
    memories: List[str]
    emotion: str
    response: str
    should_remember: bool


class NPCBrain:
    """
    LangGraph-based NPC reasoning engine.
    Graph: retrieve_memories -> reason -> generate_response -> update_memory
    """

    def __init__(
        self,
        character: CharacterPersona = MAYA,
        model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434",
        memory_store=None,
    ):
        self.character = character
        self.memory_store = memory_store
        self.llm = ChatOllama(model=model, base_url=ollama_base_url, temperature=0.8)
        self._checkpointer = MemorySaver()
        self._graph = self._build_graph()
        print(f"[NPCBrain] {character.name} initialized | model={model}")

    def _build_graph(self) -> Any:
        """Build the LangGraph reasoning graph."""
        graph = StateGraph(NPCState)
        graph.add_node("retrieve_memories", self._retrieve_memories)
        graph.add_node("reason", self._reason)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("update_memory", self._update_memory)

        graph.set_entry_point("retrieve_memories")
        graph.add_edge("retrieve_memories", "reason")
        graph.add_edge("reason", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)

        return graph.compile(checkpointer=self._checkpointer)

    def _retrieve_memories(self, state: NPCState) -> NPCState:
        """Retrieve relevant memories for context."""
        memories = []
        if self.memory_store:
            last_msg = state["messages"][-1].content if state["messages"] else ""
            memories = self.memory_store.retrieve(last_msg, session_id=state["session_id"], top_k=5)
        state["memories"] = memories
        state["emotion"] = "neutral"
        return state

    def _reason(self, state: NPCState) -> NPCState:
        """Determine emotional state and whether to remember this exchange."""
        last_msg = state["messages"][-1].content if state["messages"] else ""
        emotional_keywords = {
            "excited": ["amazing", "wow", "incredible", "love", "fantastic"],
            "thoughtful": ["wonder", "think", "perhaps", "maybe", "interesting"],
            "melancholy": ["sad", "miss", "gone", "lost", "empty"],
            "curious": ["how", "why", "what", "tell me", "explain"],
        }
        emotion = self.character.emotional_baseline.split()[0]
        for emo, keywords in emotional_keywords.items():
            if any(kw in last_msg.lower() for kw in keywords):
                emotion = emo
                break
        state["emotion"] = emotion
        state["should_remember"] = len(last_msg) > 20 and "?" not in last_msg
        return state

    def _generate_response(self, state: NPCState) -> NPCState:
        """Generate in-character response using Ollama."""
        system = self.character.system_prompt()

        memory_context = ""
        if state["memories"]:
            memory_context = "

Relevant memories from past conversations:
" +                              "
".join(f"- {m}" for m in state["memories"][:3])
            system += memory_context

        emotion_note = f"
Current emotional state: {state['emotion']}. Let this subtly color your response."
        system += emotion_note

        messages = [SystemMessage(content=system)] + state["messages"]

        try:
            response = self.llm.invoke(messages)
            state["response"] = response.content
        except Exception as e:
            state["response"] = f"*{self.character.name} pauses, lost in thought for a moment...*"
            print(f"[NPCBrain] LLM error: {e}")

        return state

    def _update_memory(self, state: NPCState) -> NPCState:
        """Store important exchanges in long-term memory."""
        if state["should_remember"] and self.memory_store:
            last_msg = state["messages"][-1].content if state["messages"] else ""
            memory_text = f"User said: '{last_msg[:100]}'. {self.character.name} felt {state['emotion']}."
            self.memory_store.store(memory_text, session_id=state["session_id"])
        return state

    def respond(self, user_message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Main entry point: takes user message, returns character response + metadata.
        """
        config = {"configurable": {"thread_id": session_id}}
        initial_state = NPCState(
            messages=[HumanMessage(content=user_message)],
            session_id=session_id,
            character_name=self.character.name,
            memories=[],
            emotion="neutral",
            response="",
            should_remember=False,
        )

        final_state = self._graph.invoke(initial_state, config=config)
        return {
            "character": self.character.name,
            "response": final_state["response"],
            "emotion": final_state["emotion"],
            "session_id": session_id,
            "memories_used": len(final_state["memories"]),
        }

    def stream_response(self, user_message: str, session_id: str = "default"):
        """Stream response tokens for real-time display."""
        system = self.character.system_prompt()
        messages = [SystemMessage(content=system), HumanMessage(content=user_message)]
        for chunk in self.llm.stream(messages):
            yield chunk.content
