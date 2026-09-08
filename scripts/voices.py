"""Audition voices and speech settings by ear.

  python scripts/voices.py                  every English voice, one file each
  python scripts/voices.py --female         just the American female ones
  python scripts/voices.py --tune af_heart  speed and pause variants of one voice
  python scripts/voices.py --say "..."      use your own line

Writes WAV files into samples/ and prints what it made. Play them, pick one,
then put it in iris/config.py as tts_voice (and tts_speed / tts_pause_ms).
"""

import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from iris.config import TTS_RATE, Config
from iris.llm import Chunker
from iris.tts import Voice

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
VOICES, TUNE = SAMPLES / "voices", SAMPLES / "tune"

# Written the way she actually talks: a short reaction, then the real sentence.
# Auditioning on a neutral sentence picks the wrong voice for this character.
LINE = ("Oh, absolutely not. That laptop of yours has been wheezing since "
        "twenty nineteen, and you know it.")

GROUPS = {
    "female": "af_",
    "male": "am_",
    "british": ("bf_", "bm_"),
}


def write_wav(path: Path, samples: np.ndarray) -> float:
    audio = np.clip(samples, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_RATE)
        w.writeframes(pcm.tobytes())
    return len(samples) / TTS_RATE


def chunked(voice: Voice, text: str, speed: float, pause_ms: int) -> np.ndarray:
    """Synthesise the way the live loop does: phrase by phrase, then joined.

    Auditioning a whole sentence in one call flatters the result -- it is not
    what you will hear. This reproduces the real seams.
    """
    chunker = Chunker()
    phrases = list(chunker.feed(text)) + list(chunker.flush())
    gap = np.zeros(int(TTS_RATE * pause_ms / 1000), dtype=np.float32)
    out = []
    for i, phrase in enumerate(phrases):
        samples, _ = voice.kokoro.create(phrase, voice=voice.cfg.tts_voice,
                                        speed=speed, lang="en-us")
        out.append(np.asarray(samples, dtype=np.float32))
        if i < len(phrases) - 1 and len(gap):
            out.append(gap)
    return np.concatenate(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--female", action="store_true")
    p.add_argument("--male", action="store_true")
    p.add_argument("--british", action="store_true")
    p.add_argument("--tune", metavar="VOICE")
    p.add_argument("--say", metavar="TEXT", default=LINE)
    args = p.parse_args()

    cfg = Config()
    # Cloning would override the timbre of every voice here with the reference,
    # which defeats the point of an audition.
    cfg.clone = False
    voice = Voice(cfg)
    # Separate directories: auditioning a voice should not delete the tuning
    # variants, and tuning should not delete the audition.
    OUT = TUNE if args.tune else VOICES
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.wav"):
        stale.unlink()

    if args.tune:
        cfg.tts_voice = args.tune
        print(f"tuning {args.tune}. lower speed sounds calmer, more pause sounds\n"
              f"more considered -- too much of either sounds sedated.\n")
        for speed in (0.95, 1.0, 1.05, 1.1):
            for pause in (0, 90, 180):
                clip = chunked(voice, args.say, speed, pause)
                name = f"{args.tune}_speed{speed}_pause{pause}.wav"
                secs = write_wav(OUT / name, clip)
                print(f"  {name:44s} {secs:.2f}s")
        print(f"\n{len(list(OUT.glob('*.wav')))} files in {OUT}")
        return 0

    names = voice.voices()
    wanted = [g for g, on in (("female", args.female), ("male", args.male),
                              ("british", args.british)) if on]
    prefixes = tuple(
        pre
        for g in (wanted or list(GROUPS))
        for pre in ((GROUPS[g],) if isinstance(GROUPS[g], str) else GROUPS[g])
    )
    picked = [n for n in names if n.startswith(prefixes)]

    print(f"writing {len(picked)} samples to {OUT}")
    print(f'saying: "{args.say}"\n')
    for name in picked:
        cfg.tts_voice = name
        clip = chunked(voice, args.say, cfg.tts_speed, 0)
        secs = write_wav(OUT / f"{name}.wav", clip)
        print(f"  {name:14s} {secs:.2f}s")

    print(f"\nplay them, then set tts_voice in iris/config.py.")
    print(f"once you have a favourite: python scripts/voices.py --tune <name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
