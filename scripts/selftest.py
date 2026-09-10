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
heard = stt(sixteen)
dt = time.perf_counter() - t
print(f"  transcribe  {dt:5.2f}s for {dur:.2f}s audio  (RTF {dt/dur:.2f})")
print(f"  heard      '{heard.text}'")
print(f"  confidence  {heard.logprob:+.2f}  "
      f"{'clear enough to remember' if heard.clear else 'answer only'}")

print("== ACTIONS ==")
# The grammar only, which is the half that has to be right. Nothing below
# touches the volume, the brightness or anything that is playing: a self-test
# that reaches for the speakers is one nobody runs twice.
from iris import actions

known = frozenset(actions.KNOWN)
should = [
    ("turn the volume down",            (actions.VOLUME, actions.DOWN)),
    ("set the volume to fifty",         (actions.VOLUME, actions.SET)),
    ("mute the sound",                  (actions.VOLUME, actions.MUTE)),
    ("dim the screen",                  (actions.BRIGHTNESS, actions.DOWN)),
    ("set the brightness to 40",        (actions.BRIGHTNESS, actions.SET)),
    ("skip this song",                  (actions.MEDIA, actions.NEXT)),
    ("pause the music",                 (actions.MEDIA, actions.PLAY)),
    ("open spotify",                    (actions.APP, actions.LAUNCH)),
    # Bare, and with the urgency adverb people actually put on the end. Both
    # used to reach the brain, which answered by inventing a mute that had not
    # happened. See "A remark is not an instruction" in the README.
    ("mute",                            (actions.VOLUME, actions.MUTE)),
    ("mute it now",                     (actions.VOLUME, actions.MUTE)),
    ("open spotify now",                (actions.APP, actions.LAUNCH)),
]
# Remarks, questions and other people's grammars. Every one of these has to
# reach the brain untouched -- a false positive here is her doing something to
# the machine because of a sentence that was about something else.
shouldnt = [
    "the screen is at fifty percent", "the volume is at maximum",
    "what's on my screen", "play chess with me", "open the pod bay doors",
    "stop", "mute your mic", "remember that I hate mushrooms",
    "set a timer for ten minutes", "I like the sound of that",
]
hits = sum(1 for t, want in should
           if (lambda a: a and (a.kind, a.op) == want)(actions.parse(t, known=known)))
miss = sum(1 for t in shouldnt if actions.parse(t, last="volume", known=known))
print(f"  recognised  {hits}/{len(should)}   (want all)")
print(f"  false hits  {miss}/{len(shouldnt)}   (want none)")
t = time.perf_counter()
for _ in range(200):
    actions.parse("turn the volume down a bit", known=known)
print(f"  parse       {(time.perf_counter()-t)/200*1000:5.3f}ms per utterance")

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
