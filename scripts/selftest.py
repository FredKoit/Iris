"""Exercise every stage without a microphone and report timings."""

import sys, time, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from iris.config import Config, FRAME, MIC_RATE, TTS_RATE

cfg = Config()
SENT = "Hello there, I am Iris, and I run entirely on this laptop."


def resample(x, src, dst):
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


print("== TTS ==")
t = time.perf_counter()
from iris.tts import Voice
v = Voice(cfg)
print(f"  load        {time.perf_counter()-t:5.2f}s")
t = time.perf_counter()
audio = v.say(SENT)
dt = time.perf_counter() - t
dur = len(audio) / TTS_RATE
print(f"  synth       {dt:5.2f}s for {dur:.2f}s audio  (RTF {dt/dur:.2f}, lower is better)")

print("== VAD ==")
from iris.vad import StreamingVAD
vad = StreamingVAD(FRAME)
sixteen = resample(audio, TTS_RATE, MIC_RATE)
probs = [vad(sixteen[i:i+FRAME]) for i in range(0, len(sixteen)-FRAME, FRAME)]
vad.reset()
sil = np.random.randn(MIC_RATE).astype(np.float32) * 0.001
sprobs = [vad(sil[i:i+FRAME]) for i in range(0, len(sil)-FRAME, FRAME)]
print(f"  speech max  {max(probs):.2f}   (want > 0.8)")
print(f"  silence max {max(sprobs):.2f}   (want < 0.3)")

print("== STT ==")
t = time.perf_counter()
from iris.stt import Transcriber
stt = Transcriber(cfg)
stt.warm()
print(f"  load        {time.perf_counter()-t:5.2f}s")
t = time.perf_counter()
text = stt(sixteen)
dt = time.perf_counter() - t
print(f"  transcribe  {dt:5.2f}s for {dur:.2f}s audio  (RTF {dt/dur:.2f})")
print(f"  heard      '{text}'")

print("== LLM ==")
from iris.llm import Brain, clean_for_speech
b = Brain(cfg)
b.warm()
cancel = threading.Event()
t = time.perf_counter()
first = None
phrases = []
for ph in b.reply("hey iris, say something rude about my laptop", cancel):
    if first is None:
        first = time.perf_counter() - t
    phrases.append(ph)
print(f"  first phrase {first:5.2f}s")
print(f"  full reply   {time.perf_counter()-t:5.2f}s")
print(f"  said        '{' '.join(phrases)}'")

print("== END TO END ==")
t = time.perf_counter()
spoken, tags = clean_for_speech(phrases[0])
_ = v.say(spoken)
print(f"  brain->audio {first + (time.perf_counter()-t):5.2f}s  (target < 1.5s)")
print(f"  emotion tags {tags}")
