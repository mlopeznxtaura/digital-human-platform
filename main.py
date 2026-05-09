"""
digital-human-platform — Entry Point

Real-time NPC system: LangGraph reasoning, Chroma memory, voice, face tracking.

Usage:
  python main.py --mode chat --character maya
  python main.py --mode server --port 8000
  python main.py --mode voice --character maya
  python main.py --mode demo
"""
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Digital Human Platform")
    parser.add_argument("--mode", required=True,
                        choices=["chat", "server", "voice", "demo"])
    parser.add_argument("--character", default="maya")
    parser.add_argument("--session", default="default")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    return parser.parse_args()


def mode_chat(args):
    """Interactive text chat with NPC."""
    from npc.brain import NPCBrain, CHARACTERS, MAYA
    from npc.memory import NPCMemoryStore

    persona = CHARACTERS.get(args.character, MAYA)
    memory = NPCMemoryStore(character_name=persona.name)
    npc = NPCBrain(character=persona, model=args.model,
                   ollama_base_url=args.ollama_url, memory_store=memory)

    print(f"
{'='*55}")
    print(f"  Digital Human: {persona.name}")
    print(f"  Model: {args.model} | Session: {args.session}")
    print(f"  Type 'quit' to exit, 'memory' to see stored memories")
    print(f"{'='*55}")

    # Greeting
    greeting_result = npc.respond(
        f"*{persona.name} notices you for the first time*",
        session_id=args.session
    )
    print(f"
{persona.name}: {greeting_result['response']}
")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"
{persona.name}: {CHARACTERS.get(args.character, MAYA).farewell if hasattr(CHARACTERS.get(args.character, MAYA), 'farewell') else 'Goodbye.'}")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            stats = memory.stats()
            print(f"[Memory] {stats}")
            continue

        result = npc.respond(user_input, session_id=args.session)
        emotion_indicator = {"happy": "😊", "melancholy": "😔", "curious": "🤔",
                              "excited": "✨", "neutral": ""}.get(result["emotion"], "")
        print(f"
{persona.name} {emotion_indicator}: {result['response']}
")


def mode_server(args):
    """Start FastAPI + Socket.io server."""
    import uvicorn
    from api.server import socket_app
    print(f"[Server] Starting on {args.host}:{args.port}")
    uvicorn.run(socket_app, host=args.host, port=args.port, log_level="info")


def mode_voice(args):
    """Voice interaction — microphone in, speaker out."""
    from npc.brain import NPCBrain, CHARACTERS, MAYA
    from npc.memory import NPCMemoryStore
    from npc.voice import WhisperASR, CoquiTTSVoice

    persona = CHARACTERS.get(args.character, MAYA)
    memory = NPCMemoryStore(character_name=persona.name)
    npc = NPCBrain(character=persona, model=args.model, memory_store=memory)
    asr = WhisperASR(model_size="base")

    try:
        tts = CoquiTTSVoice(language="en")
        tts_available = True
    except Exception as e:
        print(f"[Voice] TTS unavailable: {e}")
        tts_available = False

    print(f"
[Voice] {persona.name} is listening... (Ctrl+C to quit)")
    while True:
        try:
            transcript = asr.listen_once(max_duration=10.0)
            if not transcript:
                continue
            print(f"You: {transcript}")
            result = npc.respond(transcript, session_id=args.session)
            print(f"{persona.name}: {result['response']}")
            if tts_available:
                tts.speak(result["response"], play_audio=True)
        except KeyboardInterrupt:
            print("
Goodbye.")
            break


def mode_demo(args):
    """Run a canned demo conversation."""
    from npc.brain import NPCBrain, MAYA
    from npc.memory import NPCMemoryStore

    memory = NPCMemoryStore(character_name=MAYA.name)
    npc = NPCBrain(character=MAYA, model=args.model, memory_store=memory)

    exchanges = [
        "Hi there. Who are you?",
        "What made you leave astrophysics?",
        "Do you miss the stars?",
        "Tell me something about consciousness.",
    ]

    print(f"
{'='*55}")
    print(f"  Digital Human Demo: {MAYA.name}")
    print(f"{'='*55}
")

    for msg in exchanges:
        print(f"User: {msg}")
        result = npc.respond(msg, session_id="demo")
        print(f"{MAYA.name}: {result['response']}")
        print(f"  [emotion: {result['emotion']} | memories used: {result['memories_used']}]
")


def main():
    args = parse_args()
    print("=" * 55)
    print("  Digital Human Platform")
    print(f"  Mode: {args.mode.upper()} | Character: {args.character}")
    print("=" * 55)

    dispatch = {
        "chat": mode_chat,
        "server": mode_server,
        "voice": mode_voice,
        "demo": mode_demo,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
