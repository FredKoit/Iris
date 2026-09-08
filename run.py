"""Entry point. `python run.py` for voice, `python run.py --text` to type."""

import argparse
import sys

import ollama
import sounddevice as sd

from iris.app import Iris
from iris.config import Config


def preflight(cfg: Config) -> bool:
    try:
        tags = ollama.Client(host=cfg.ollama_host).list()
    except Exception as e:
        print(f"cannot reach ollama at {cfg.ollama_host}: {e}")
        print("start it with:  ollama serve")
        return False
    names = sorted(m.model for m in tags.models)
    want = cfg.ollama_model

    # An exact tag must never resolve to a different one. Matching on the base
    # name alone silently ran "qwen3:4b-instruct-2507-q4_K_M" as "qwen3:4b" --
    # a different model with different behaviour -- because both start "qwen3".
    # Only a name with no tag at all may match by base.
    if want in names:
        match = want
    elif f"{want}:latest" in names:
        match = f"{want}:latest"
    elif ":" not in want:
        match = next((n for n in names if n.split(":")[0] == want), None)
    else:
        match = None

    if match is None:
        print(f"model '{want}' not pulled. run:")
        print(f"  ollama pull {want}")
        print(f"have: {', '.join(names) or '(none)'}")
        return False
    cfg.ollama_model = match
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Iris - a local voice AI")
    p.add_argument("--text", action="store_true", help="type instead of speaking")
    p.add_argument("--model", help="override the ollama model")
    p.add_argument("--voice", help="override the kokoro voice")
    p.add_argument("--whisper", help="tiny.en / base.en / small.en")
    p.add_argument("--barge-in", action="store_true",
                   help="let her be interrupted mid-sentence (headphones only)")
    p.add_argument("--no-idle", action="store_true",
                   help="never speak first; only answer when spoken to")
    p.add_argument("--no-tray", action="store_true",
                   help="no tray icon and no global hotkeys")
    p.add_argument("--no-screen", action="store_true",
                   help="do not tell her which application is in front")
    p.add_argument("--push-to-talk", action="store_true",
                   help="only listen while the talk key is held")
    p.add_argument("--no-timers", action="store_true",
                   help="do not act on \"remind me in ten minutes\"")
    p.add_argument("--idle-after", type=float, metavar="SECONDS",
                   help="silence before she says something unprompted")
    p.add_argument("--input-device", type=int)
    p.add_argument("--output-device", type=int)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    cfg = Config()
    if args.model:
        cfg.ollama_model = args.model
    if args.voice:
        cfg.tts_voice = args.voice
    if args.whisper:
        cfg.whisper_model = args.whisper
    if args.barge_in:
        cfg.barge_in = True
    if args.no_idle:
        cfg.idle = False
    if args.no_tray:
        cfg.tray = False
    if args.no_screen:
        cfg.screen = False
    if args.push_to_talk:
        cfg.push_to_talk = True
    if args.no_timers:
        cfg.timers = False
    if args.idle_after is not None:
        cfg.idle_after_s = args.idle_after
    if args.input_device is not None:
        cfg.input_device = args.input_device
    if args.output_device is not None:
        cfg.output_device = args.output_device

    if args.list_voices:
        from iris.tts import Voice
        print("\n".join(Voice(cfg).voices()))
        return 0

    if not preflight(cfg):
        return 1

    iris = Iris(cfg)
    # Push-to-talk is a global hotkey, so it needs the same machinery as the
    # tray even when the icon itself is switched off.
    if cfg.tray or cfg.push_to_talk:
        from iris.tray import attach
        attach(iris, cfg)
    if args.text:
        iris.run_text()
    else:
        iris.run_voice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
