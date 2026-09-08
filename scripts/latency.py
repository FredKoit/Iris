"""Measure a full turn the way the ear experiences it: from the moment you
stop talking to the moment she starts."""

import sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iris.config import Config, MIC_RATE, TTS_RATE
from iris.llm import Brain, clean_for_speech
from iris.stt import Transcriber
from iris.tts import Voice

TURNS = [
    "hey iris, what are you doing?",
    "do you like living on a laptop?",
    "what's your favourite colour and why?",
    "tell me something i don't know",
]

cfg = Config()
voice = Voice(cfg)
stt = Transcriber(cfg); stt.warm()
brain = Brain(cfg); brain.warm()


def as_mic(text):
    """Fake the user's voice so the numbers include real transcription work."""
    a, _ = voice.kokoro.create(text, voice="am_michael", speed=1.0)
    a = np.asarray(a, dtype=np.float32)
    n = int(len(a) * MIC_RATE / TTS_RATE)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)

print(f"{'turn':34s} {'stt':>6s} {'llm':>6s} {'tts':>6s} {'total':>7s}  (+500ms endpoint)")
totals = []
for t_text in TURNS:
    audio = as_mic(t_text)

    t = time.perf_counter(); heard = stt(audio); stt_s = time.perf_counter() - t

    cancel = threading.Event()
    t = time.perf_counter(); gen = brain.reply(heard or t_text, cancel)
    first = next(gen); llm_s = time.perf_counter() - t

    spoken, _ = clean_for_speech(first)
    t = time.perf_counter(); wav = voice.say(spoken); tts_s = time.perf_counter() - t
    for _ in gen:
        pass

    total = stt_s + llm_s + tts_s
    totals.append(total)
    print(f"{t_text[:33]:34s} {stt_s:5.2f}s {llm_s:5.2f}s {tts_s:5.2f}s {total:6.2f}s")
    print(f"{'':34s} -> \"{spoken}\" ({len(wav)/TTS_RATE:.1f}s of audio)")

print(f"\nmean time to first audio: {sum(totals)/len(totals):.2f}s"
      f"  (+0.50s endpoint = {sum(totals)/len(totals)+0.5:.2f}s perceived)")
