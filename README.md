# Digital Human Platform

Cluster 16 of the NextAura 500 SDKs / 25 Clusters project.

Real-time generative AI NPCs with voice, lip-synced face, and persistent memory. Generative AI characters that reason, remember, and speak with genuine personality.

## Architecture

- LangGraph for stateful character reasoning decision graph
- LlamaIndex + Chroma for long-term episodic memory
- Ollama for local LLM inference (Mistral, Llama-3, etc.)
- Whisper for real-time speech-to-text
- Coqui XTTS for voice cloning / character TTS
- NVIDIA Riva for production-grade ASR/TTS
- NVIDIA Audio2Face for lip-sync facial animation
- MediaPipe for user face tracking and gesture detection
- LiveKit for WebRTC audio/video streaming
- React Three Fiber for 3D avatar rendering in browser
- FastAPI + Socket.io for real-time session management
- Redis + Supabase for session state and memory persistence

## SDKs Used

NVIDIA ACE SDK, NVIDIA Audio2Face, NVIDIA Riva SDK, Coqui TTS, OpenAI Whisper, LangGraph, LlamaIndex, Chroma SDK, Ollama, SAM2, MediaPipe SDK, LiveKit SDK, WebRTC SDK, React Three Fiber, Three.js, Socket.io SDK, Supabase SDK, FastAPI, Redis SDK, Pydantic AI

## Quickstart

```bash
pip install -r requirements.txt
ollama pull mistral  # pull a local LLM

# Start the NPC server
python main.py --mode server --character maya

# Chat with the NPC
python main.py --mode chat --character maya --session user123

# Run with voice (requires microphone + speakers)
python main.py --mode voice --character maya

# Launch full avatar demo
python main.py --mode demo
```

## Structure

```
npc/
  brain.py         LangGraph reasoning graph — core NPC decision loop
  memory.py        LlamaIndex + Chroma episodic memory
  persona.py       Character persona definitions and state
  voice.py         Whisper ASR + Coqui/Riva TTS pipeline
perception/
  face_tracker.py  MediaPipe face mesh + emotion detection
  gesture.py       Hand gesture recognition
animation/
  audio2face.py    NVIDIA Audio2Face lip-sync client
  avatar_sync.py   WebSocket bridge to React Three Fiber avatar
api/
  server.py        FastAPI + Socket.io NPC interaction API
  session.py       Redis session state management
characters/        Character persona JSON configs
main.py            Entry point
```
