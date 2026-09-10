"""Set up and test the VTube Studio avatar.

  python scripts/avatar.py                 what the model exposes, and the mapping
  python scripts/avatar.py --sweep         show every expression in turn, so you
                                           can see what each does and name it
  python scripts/avatar.py --try heart      show one expression (or hotkey)
  python scripts/avatar.py --mouth         mouth-only lip sync test, no speech
  python scripts/avatar.py --probe         what each VTS input actually moves on
                                           this model, and which are inverted
  python scripts/avatar.py --eyes          gaze and blink, on the avatar and as
                                           a trace you can read; --talk drives it
                                           as if speaking, --dry needs no VTS
  python scripts/avatar.py --say "hello"   speak a line with the mouth following

VTube Studio must be running with the API started (gear icon, port 8001).
Once you know which expression is which, edit `vtube_map` in iris/config.py.
It is keyed by emotion cue and holds the expression's name.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.config import Config
from iris.vtube import VTubeStudio


def show(vts: VTubeStudio, cfg: Config) -> None:
    print(f"\nmodel: {vts.model}")
    reverse = {f: tag for tag, f in vts.expressions.items()}
    print(f"\n{len(vts.available)} expressions (driven by explicit on/off):")
    for e in vts.available:
        tag = reverse.get(e["file"])
        print(f"   {e['name'] or '(unnamed)':14s} {e['file']}"
              + (f"   <- [{tag}]" if tag else ""))

    others = [h for h in vts.all_hotkeys if h["type"] != "ToggleExpression"]
    if others:
        print(f"\n{len(others)} other hotkeys (animations -- fire with --try):")
        for h in others:
            print(f"   {h['name'] or '(unnamed)':14s} {h['type']:18s} {h['file']}")

    print("\nemotion cues:")
    for tag in cfg.vtube_emotions:
        wanted = cfg.vtube_map.get(tag)
        if tag in vts.expressions:
            print(f"   [{tag}]".ljust(16) + f"-> {vts.expressions[tag]}")
        elif tag == "neutral":
            print(f"   [{tag}]".ljust(16) + "-> clears whatever is showing")
        else:
            miss = f"'{wanted}' not found" if wanted else "nothing matched"
            print(f"   [{tag}]".ljust(16) + f"-- {miss}")


_PROBE_INPUTS = ("FaceAngleX", "FaceAngleY", "FaceAngleZ",
                 "FacePositionX", "FacePositionY", "FacePositionZ",
                 "EyeLeftX", "EyeLeftY", "EyeOpenLeft",
                 "MouthOpen", "MouthSmile", "MouthX",
                 "Brows", "BrowLeftY", "BrowRightY", "CheekPuff", "TongueOut")


def probe(vts: VTubeStudio) -> None:
    """Inject each input parameter in turn and report what it actually moves.

    Which VTS input reaches which rig parameter is set per model, in the VTS
    interface, and guessing it wrong is silent -- the plugin keeps sending
    values that land nowhere. Two things this catches are worth the round trips:
    inputs that drive nothing at all, and inputs whose sign is inverted.

    The second is the one that matters. `vtube_gaze_flip_x` and `_y` exist
    because a head parameter may point the opposite way to the eye one, and on
    the model this was written against the horizontal pair does while the
    vertical pair does not. Left wrong, her head turns away from everything she
    looks at.
    """
    from iris.motion import IdleMotion

    rest = {k: 0.0 for k in _PROBE_INPUTS}
    rest.update({"EyeOpenLeft": 1.0, "EyeOpenRight": 1.0})

    def rig() -> dict:
        data = vts._send("Live2DParameterListRequest", {}) or {}
        return {p["name"]: p["value"] for p in data.get("parameters", [])}

    def hold(values: dict, seconds: float) -> dict:
        """Keep injecting for `seconds`, then read the rig.

        Injecting once and waiting does not work: VTS lets a parameter fall
        back to its tracking default as soon as the plugin stops sending it, so
        a single injection is already decaying by the time it is read. The live
        loop sends every frame, and so must this or it measures the way back
        down rather than the value.
        """
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            vts.inject(values)
            time.sleep(0.03)
        vts.inject(values)
        return rig()

    limits = {}
    listed = vts._send("InputParameterListRequest", {}) or {}
    for p in listed.get("defaultParameters", []):
        limits[p.get("name")] = (p.get("min", -1.0), p.get("max", 1.0))

    # Anything that moves while nothing is being injected is on a physics
    # simulation -- dangling hair, an accessory, a swinging prop. Those drift
    # far more than the rig parameters do and would otherwise fill every row
    # and pick up meaningless INVERTED marks. Found rather than hardcoded,
    # since every model names its trinkets differently.
    first = hold(rest, 0.6)
    second = hold(rest, 1.0)
    noisy = {k for k, v in second.items() if abs(v - first.get(k, 0.0)) > 0.02}

    print(f"\nmodel: {vts.model}")
    if noisy:
        print(f"ignoring {len(noisy)} parameters that move on their own "
              f"(physics): {', '.join(sorted(noisy)[:6])}"
              + (", ..." if len(noisy) > 6 else ""))
    print()
    print(f"{'inject':15}{'value':>8}   moves")
    print("-" * 72)
    for name in _PROBE_INPUTS:
        low, high = limits.get(name, (-1.0, 1.0))
        value = high * 0.8 if high > 0 else low * 0.8
        # Reset everything between probes and re-read the baseline each time.
        # Injected values persist until overwritten, so without this the tail
        # of one probe is still settling into the next one's reading.
        before = hold(rest, 0.5)
        settling = hold({**rest, name: value}, 0.5)
        after = hold({**rest, name: value}, 0.4)
        # Moved, and then stopped moving. A driven parameter converges and
        # holds; physics is still swinging on the second read. The tolerance
        # has to be a fraction of the travel rather than a fixed number --
        # ParamAngleX swings 28 units and settles to within a few tenths, which
        # any absolute threshold tight enough for a 0..1 parameter rejects.
        moved = []
        for k, value_now in after.items():
            if k in noisy:
                continue
            travel = value_now - before.get(k, 0.0)
            if abs(travel) <= 0.05:
                continue
            if abs(value_now - settling.get(k, 0.0)) > 0.1 * abs(travel):
                continue
            moved.append((k, travel))
        moved.sort(key=lambda kv: -abs(kv[1]))
        if not moved:
            print(f"{name:15}{value:>8.1f}   -- nothing --")
            continue
        parts = []
        for k, delta in moved[:3]:
            sign = "" if delta * value > 0 else "  <- INVERTED"
            parts.append(f"{k} {delta:+.2f}{sign}")
        print(f"{name:15}{value:>8.1f}   " + ", ".join(parts))

    vts.inject({**IdleMotion(Config()).rest(), "MouthOpen": 0.0})
    print("\nAn input that moves nothing is one this model does not map, so "
          "whatever\nfeature drives it does nothing here. A head parameter "
          "inverted against a\nnon-inverted eye one (or the reverse) is what "
          "vtube_gaze_flip_x and _y are\nfor -- per axis, because a rig can "
          "disagree on one and agree on the other.")


def eyes(cfg: Config, vts: VTubeStudio | None, seconds: float, talk: bool) -> None:
    """Run the idle motion and draw where the eyes are going.

    The point is the shape of it, which numbers do not show: her eyes should sit
    still and then flick, never slide. Each row is one frame -- a run of rows at
    the same place is a fixation, a row that lands somewhere else is a saccade,
    and B is a blink. On a big enough jump the head should start moving a beat
    *after* the eyes, and the eyes should then ease back towards centre as it
    catches up. If the head instead pulls away from where she just looked, flip
    vtube_gaze_flip_x in the config; --probe measures which way the rig goes.
    """
    from iris.motion import IdleMotion, GAZE_X, GAZE_Y, EYES_OPEN, HEAD_YAW

    motion = IdleMotion(cfg)
    period = 1.0 / cfg.vtube_lipsync_fps
    width = 51
    half = width // 2
    print("\n" + " " * 32 + "left".ljust(half - 3) + "centre" + "right".rjust(half - 3))
    print("     t   eyeX   eyeY  head  lid " + "-" * half + "+" + "-" * half)
    started = time.perf_counter()
    while (t := time.perf_counter() - started) < seconds:
        tick = time.perf_counter()
        # A syllable-rate mouth, so the speech couplings are visible: while she
        # is talking she should look away more, hold each point for less time,
        # and blink about twice as often.
        level = max(0.0, 0.55 + 0.45 * math.sin(t * 14.0)) if talk else 0.0
        frame = motion.frame(t, level)
        if vts is not None:
            frame["MouthOpen"] = level
            vts.inject(frame)

        gx, gy, lid = frame[GAZE_X[0]], frame[GAZE_Y[0]], frame[EYES_OPEN[0]]
        col = int((gx + 1.0) / 2.0 * (width - 1))
        mark = "B" if lid < 0.4 else ("o" if lid < 0.9 else "*")
        row = [" "] * width
        row[half] = "|"
        row[max(0, min(width - 1, col))] = mark
        print(f"{t:6.2f} {gx:+.3f} {gy:+.3f} {frame[HEAD_YAW]:+5.1f} {lid:.2f} "
              + "".join(row), flush=True)
        time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    if vts is not None:
        vts.inject({**motion.rest(), "MouthOpen": 0.0})
    print("\n* eyes open   o lids lowered (looking down, or mid-blink)   B blink")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--try", dest="fire", metavar="NAME")
    p.add_argument("--mouth", action="store_true")
    p.add_argument("--probe", action="store_true",
                   help="which VTS input drives which rig parameter, and which "
                        "are inverted")
    p.add_argument("--eyes", action="store_true",
                   help="run the gaze and blink model, on the avatar and as a trace")
    p.add_argument("--talk", action="store_true",
                   help="with --eyes: drive it as if she were speaking")
    p.add_argument("--dry", action="store_true",
                   help="with --eyes: trace only, no VTube Studio needed")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="how long --eyes runs")
    p.add_argument("--say", metavar="TEXT")
    p.add_argument("--pause", type=float, default=3.0,
                   help="seconds each expression is held while sweeping")
    args = p.parse_args()

    cfg = Config()
    if args.eyes and args.dry:
        eyes(cfg, None, args.seconds, args.talk)
        return 0

    vts = VTubeStudio(cfg)
    if not vts.connect():
        print("could not connect. Is VTube Studio running with the API started?")
        if args.eyes:
            print("(--eyes --dry runs the trace without it)")
        return 1

    try:
        if args.fire:
            if vts.trigger(args.fire):
                print(f"fired hotkey {args.fire!r}")
            else:
                # Fall back to expression names. A hotkey is often labelled
                # differently from the expression it toggles ("glass[1]" for
                # megane.exp3.json), and it is the expression name that goes
                # into vtube_map -- so that is the name someone will type here.
                match = next(
                    (e["file"] for e in vts.available
                     if args.fire.strip().lower() in (e["name"].lower(),
                                                      e["file"].lower())),
                    None,
                )
                if match is None:
                    print(f"no hotkey or expression called {args.fire!r}")
                else:
                    vts._activate(match, True)
                    time.sleep(args.pause)
                    vts._activate(match, False)
                    print(f"showed expression {args.fire!r} for {args.pause}s")

        elif args.sweep:
            # Expressions first, and driven the same way Iris drives them:
            # explicit on, then off. Firing their hotkeys instead would miss any
            # expression that has no hotkey, and would leave whatever it
            # toggled switched on afterwards.
            print(f"\nshowing each expression for {args.pause}s. Watch the avatar\n"
                  f"and note what each one does -- then map the names into\n"
                  f"cfg.vtube_map.\n")
            for e in vts.available:
                label = e["name"] or e["file"]
                print(f"   {label}", flush=True)
                vts._activate(e["file"], True)
                time.sleep(args.pause)
                vts._activate(e["file"], False)

            others = [h for h in vts.all_hotkeys if h["type"] != "ToggleExpression"]
            if others:
                print(f"\nand {len(others)} animation hotkeys:\n")
                for h in others:
                    label = h["name"] or h["file"] or "(unnamed)"
                    print(f"   {label}  [{h['type']}]", flush=True)
                    vts._send("HotkeyTriggerRequest", {"hotkeyID": h["id"]})
                    time.sleep(args.pause)

            print("\ndone. Her face is back to neutral.")

        elif args.mouth:
            print("mouth sweep for 6s, watch the avatar...")
            t0 = time.time()
            while time.time() - t0 < 6:
                vts.set_mouth((math.sin((time.time() - t0) * 3.0) + 1) / 2)
                time.sleep(1 / cfg.vtube_lipsync_fps)
            vts.set_mouth(0.0)
            print("done")

        elif args.probe:
            probe(vts)

        elif args.eyes:
            eyes(cfg, None if args.dry else vts, args.seconds, args.talk)

        elif args.say:
            from iris.player import Player
            from iris.tts import Voice
            print("loading voice...")
            voice = Voice(cfg)
            player = Player(cfg)
            player.start()
            vts.start_lipsync(player)
            player.play(voice.say(args.say))
            while player.busy:
                time.sleep(0.02)
            time.sleep(0.3)
            vts.set_mouth(0.0)
            player.close()
            print("done")

        else:
            show(vts, cfg)

    finally:
        vts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
