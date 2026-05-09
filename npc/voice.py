"""
Voice pipeline: Whisper ASR + Coqui XTTS TTS for NPC voice interaction.
Full duplex: listen to user, respond in character voice.
SDKs: OpenAI Whisper, Coqui TTS, PyAudio, soundfile
"""
import os
import io
import time
import queue
import threading
import tempfile
from typing import Optional, Callable, Generator
from pathlib import Path

import numpy as np
import soundfile as sf
import whisper
import torch

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False
    print("Warning: Coqui TTS not available. Install: pip install TTS")


class WhisperASR:
    """
    Real-time speech recognition using OpenAI Whisper.
    Captures microphone audio, transcribes in sliding windows.
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    SILENCE_THRESHOLD = 0.02
    SILENCE_DURATION = 1.5      # seconds of silence = end of utterance

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Loading {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        print("[Whisper] ASR ready")

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file."""
        result = self.model.transcribe(audio_path, language=language, fp16=(self.device == "cuda"))
        return result["text"].strip()

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM audio bytes."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = self.model.transcribe(audio, fp16=(self.device == "cuda"))
        return result["text"].strip()

    def _is_silent(self, audio_chunk: np.ndarray) -> bool:
        return np.abs(audio_chunk).mean() < self.SILENCE_THRESHOLD

    def listen_once(self, max_duration: float = 10.0) -> Optional[str]:
        """
        Listen for a single utterance from microphone.
        Returns transcription when silence detected.
        """
        if not PYAUDIO_AVAILABLE:
            print("[Whisper] PyAudio not available — cannot listen from microphone")
            return None

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        frames = []
        silence_frames = 0
        silence_limit = int(self.SILENCE_DURATION * self.SAMPLE_RATE / self.CHUNK_SIZE)
        max_frames = int(max_duration * self.SAMPLE_RATE / self.CHUNK_SIZE)
        recording = False

        print("[Whisper] Listening... (speak now)")
        for _ in range(max_frames):
            data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if not self._is_silent(chunk):
                recording = True
                silence_frames = 0
            elif recording:
                silence_frames += 1
                if silence_frames >= silence_limit:
                    break

            if recording:
                frames.append(data)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        if not frames:
            return None

        audio_bytes = b"".join(frames)
        return self.transcribe_bytes(audio_bytes)


class CoquiTTSVoice:
    """
    Character TTS voice using Coqui XTTS v2.
    Clone any voice, synthesize in 17+ languages.
    """

    def __init__(
        self,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        device: str = "cuda",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    ):
        if not COQUI_AVAILABLE:
            raise ImportError("Coqui TTS required. Install: pip install TTS")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.speaker_wav = speaker_wav
        self.language = language
        print(f"[CoquiTTS] Loading XTTS v2 on {self.device}...")
        self.tts = TTS(model_name).to(self.device)
        print("[CoquiTTS] Voice ready")

    def speak(
        self,
        text: str,
        output_path: Optional[str] = None,
        play_audio: bool = True,
    ) -> str:
        """Synthesize text to speech and optionally play it."""
        output_path = output_path or tempfile.mktemp(suffix=".wav")
        self.tts.tts_to_file(
            text=text,
            speaker_wav=self.speaker_wav,
            language=self.language,
            file_path=output_path,
        )
        if play_audio and PYAUDIO_AVAILABLE:
            self._play_wav(output_path)
        return output_path

    def speak_stream(self, text: str) -> Generator[bytes, None, None]:
        """Stream audio chunks for real-time playback via WebSocket."""
        wav_path = self.speak(text, play_audio=False)
        with open(wav_path, "rb") as f:
            while chunk := f.read(4096):
                yield chunk

    def _play_wav(self, wav_path: str):
        """Play a WAV file through speakers."""
        data, sr = sf.read(wav_path)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=1 if data.ndim == 1 else data.shape[1],
            rate=sr,
            output=True,
        )
        stream.write(data.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()
        pa.terminate()
