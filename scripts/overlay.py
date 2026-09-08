"""Float the VTube Studio model on the desktop, over everything else.

  python scripts/overlay.py            float her: transparent, borderless, on top
  python scripts/overlay.py --click    ...and let clicks pass through to whatever
                                       is behind her
  python scripts/overlay.py --keep     ...and keep putting her back on top,
                                       which VTube Studio otherwise undoes
  python scripts/overlay.py --restore  put the window back to normal
  python scripts/overlay.py --key 00FF00   force the background colour to key out
  python scripts/overlay.py --move 1200 400 --size 500 700

Nothing is captured or re-rendered: VTube Studio's own window is restyled so
Windows keys its background colour out. That costs nothing at runtime and stays
in sync with the model for free.

In VTube Studio first: set a *solid* background colour, and hide the interface
(it is drawn inside the same window, so it stays visible otherwise). The colour
is detected as the most common one in the window, or given with --key. Anything
in the model that is exactly the key colour goes transparent too, so pick one
the model does not use.

The window handling itself lives in iris/overlay.py, so the tray menu and the
global hotkeys can drive the same code.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.overlay import (SWP_SHOWWINDOW, apply, background_colour,
                          ensure_topmost, find_window, rect, restore,
                          start_keeper, unkey, user32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    p.add_argument("--click", action="store_true",
                   help="let mouse clicks pass through to what is behind her")
    p.add_argument("--key", metavar="RRGGBB",
                   help="background colour to make transparent (default: detected)")
    p.add_argument("--move", nargs=2, type=int, metavar=("X", "Y"))
    p.add_argument("--size", nargs=2, type=int, metavar=("W", "H"))
    p.add_argument("--keep", action="store_true",
                   help="stay running and put her back on top when VTS drops it")
    args = p.parse_args()

    hwnd = find_window()
    if hwnd is None:
        print("VTube Studio window not found. Is it running?")
        return 1
    x, y, w, h = rect(hwnd)
    print(f"VTube Studio: hwnd={hwnd} at {x},{y} size {w}x{h}")

    if args.restore:
        restore(hwnd)
        print("restored: normal window, not on top, no transparency")
        return 0

    if args.move or args.size:
        nx, ny = args.move if args.move else (x, y)
        nw, nh = args.size if args.size else (w, h)
        user32.SetWindowPos(hwnd, 0, nx, ny, nw, nh, SWP_SHOWWINDOW)
        x, y, w, h = rect(hwnd)
        print(f"moved to {x},{y} size {w}x{h}")

    if args.key:
        rgb = int(args.key.lstrip("#"), 16)
        # COLORREF is 0x00BBGGRR, the reverse of the way colours are written.
        key = ((rgb & 0xFF) << 16) | (rgb & 0xFF00) | ((rgb >> 16) & 0xFF)
    else:
        # Show the window's real pixels, and put it in front so nothing overlaps
        # the area being sampled. Topmost as well as foreground: if the window
        # has been demoted, SetForegroundWindow alone can leave another app
        # covering it, and the sample then measures that app's colour instead.
        unkey(hwnd)
        ensure_topmost(hwnd)
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.4)
        key = background_colour(hwnd, report=True)
    print(f"keying out colour: #{key & 0xFF:02X}{(key >> 8) & 0xFF:02X}"
          f"{(key >> 16) & 0xFF:02X} (BGR 0x{key:06X})")

    apply(hwnd, key, args.click)
    print("floating: borderless, always on top"
          + (", clicks pass through" if args.click else ""))
    print("undo with: python scripts/overlay.py --restore")

    if not args.keep:
        # Worth saying plainly: VTube Studio puts its own window styles back,
        # and when the layered flag goes the colour key goes with it, so the
        # background reappears. Applying this once does not stay applied.
        print("\nnote: VTube Studio will undo this on its own. Keep it applied"
              "\n      with `python run.py`, or `python scripts/overlay.py --keep`.")

    if args.keep:
        # VTube Studio re-asserts its own z-order, so topmost has to be put
        # back rather than simply set. `python run.py` does this on its own;
        # this is for keeping her up without her running.
        print("\nkeeping her on top. ctrl+c to stop.")
        start_keeper()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nstopped. She will drop behind other windows now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
