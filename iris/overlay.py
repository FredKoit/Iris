"""Restyle the VTube Studio window so she floats on the desktop.

Nothing is captured or re-rendered: VTube Studio's own window is restyled so
Windows keys its background colour out. That costs nothing at runtime and stays
in sync with the model for free.

This is the mechanism only. `scripts/overlay.py` is the command line around it,
and `iris/tray.py` reaches in for the click-through toggle, which is the one
control worth having on a hotkey -- it decides whether she is a thing on your
desktop or a thing in the way of it.

Windows only: it is all user32/gdi32.
"""

import ctypes
import ctypes.wintypes as wt
import json
import threading

from .config import ROOT

STATE = ROOT / ".overlay_state.json"

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_CAPTION, WS_THICKFRAME = 0x00C00000, 0x00040000
WS_MINIMIZEBOX, WS_MAXIMIZEBOX, WS_SYSMENU = 0x00020000, 0x00010000, 0x00080000
WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x00080000, 0x20, 0x80
WS_EX_TOPMOST = 0x00000008
LWA_COLORKEY = 0x1
SWP_NOMOVE, SWP_NOSIZE, SWP_FRAMECHANGED, SWP_SHOWWINDOW = 0x2, 0x1, 0x20, 0x40
SWP_NOACTIVATE = 0x0010

# Every signature below is declared on purpose, and the pointer-sized ones are
# why. Left undeclared, ctypes passes each argument as a C int: HWND_TOPMOST
# (-1) reaches a 64-bit HWND parameter as 0x00000000FFFFFFFF instead of all
# ones, so SetWindowPos rejects it and returns false -- which looks exactly
# like "always on top does not work". The same truncation silently cuts the
# handle GetDC hands back, which is what made colour sampling unreliable.
user32.SetLayeredWindowAttributes.argtypes = [wt.HWND, wt.COLORREF, ctypes.c_byte,
                                              wt.DWORD]
user32.SetLayeredWindowAttributes.restype = wt.BOOL
user32.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_uint]
user32.SetWindowPos.restype = wt.BOOL
user32.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetDC.argtypes = [wt.HWND]
user32.GetDC.restype = wt.HDC
user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.IsWindowVisible.argtypes = [wt.HWND]
gdi32.GetPixel.argtypes = [wt.HDC, ctypes.c_int, ctypes.c_int]
gdi32.GetPixel.restype = wt.COLORREF

# Sign-extended to the full pointer width, for the same reason.
HWND_TOPMOST = wt.HWND(-1)
HWND_NOTOPMOST = wt.HWND(-2)

_REFRESH = SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED | SWP_SHOWWINDOW


def find_window() -> int | None:
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        # Match on class as well as title: the title alone also matches an editor
        # tab or a browser page that happens to mention VTube Studio.
        if cls.value == "UnityWndClass" and "vtube studio" in buf.value.lower():
            found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def rect(hwnd) -> tuple[int, int, int, int]:
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


CLR_INVALID = 0xFFFFFFFF


def _sample(dc, ox: int, oy: int, w: int, h: int, steps: int = 12):
    """Grid-sample a device context. Sampling one corner is not enough: with
    the VTube Studio interface showing, the corners are its dark chrome and the
    background sits in the middle."""
    import collections

    seen = collections.Counter()
    for i in range(1, steps):
        for j in range(1, steps):
            px = gdi32.GetPixel(dc, ox + w * i // steps, oy + h * j // steps)
            if px != CLR_INVALID:
                seen[px] += 1
    return seen


def background_colour(hwnd, report: bool = False) -> int:
    """The most common colour in the window: whatever the background is.

    Read from the window's own device context rather than from the screen.
    Reading the screen means reading whatever happens to be stacked on top of
    her, and Windows refuses SetForegroundWindow to a background process, so
    there is no reliable way to clear the way first. That is not a rare edge
    case: run this from a terminal and the terminal is usually the thing in
    front, so the sample measured an editor's dark grey and keyed *that* out,
    leaving a green backdrop perfectly intact.

    The window DC has her pixels whatever the z-order. A GPU-rendered window
    can refuse to hand them over, so the screen remains the fallback.
    """
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    seen = None
    dc = user32.GetDC(hwnd)
    if dc:
        try:
            seen = _sample(dc, 0, 0, r.right, r.bottom)
        finally:
            user32.ReleaseDC(hwnd, dc)

    where = "the window"
    if not seen:
        where = "the screen"
        x, y, w, h = rect(hwnd)
        dc = user32.GetDC(None)
        try:
            seen = _sample(dc, x, y, w, h)
        finally:
            user32.ReleaseDC(None, dc)

    top, count = seen.most_common(1)[0]
    if report:
        share = 100 * count / sum(seen.values())
        print(f"sampled {sum(seen.values())} points from {where}: "
              f"most common colour covers {share:.0f}%")
        if share < 25:
            print("  warning: no colour dominates. Is a solid background set,")
            print("  and is the VTube Studio interface hidden?")
    return top


def unkey(hwnd) -> None:
    """Drop transparency so the window shows its own pixels again.

    Detection reads the screen, so with a key already applied it would measure
    whatever is *behind* her and key that out instead -- keying the colour of
    the editor window in the background rather than her backdrop.
    """
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                          ex & ~(WS_EX_LAYERED | WS_EX_TRANSPARENT))
    user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, _REFRESH)


_DECORATION = (WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX
               | WS_MAXIMIZEBOX | WS_SYSMENU)


def state() -> dict:
    """The saved original window styles, and the overlay we want on top of it."""
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def apply(hwnd, key: int, click_through: bool) -> None:
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    saved = state()
    # The original styles are recorded once and never overwritten -- by the
    # second call the window is already borderless, and saving that as the
    # "original" would make --restore a no-op forever.
    saved.setdefault("style", style)
    saved.setdefault("exstyle", exstyle)
    # ...but what we *want* is recorded every time, because the keeper below
    # has to know what to put back.
    saved["key"] = key
    saved["click_through"] = bool(click_through)
    STATE.write_text(json.dumps(saved), encoding="utf-8")
    ensure_overlay(hwnd, key, click_through, force=True)


def ensure_overlay(hwnd, key: int, click_through: bool,
                   force: bool = False) -> bool:
    """Put back whatever VTube Studio has undone. True if it had to.

    VTS re-applies its own window styles -- measured, the title bar and the
    layered flag both came back on their own -- and losing WS_EX_LAYERED also
    discards the colour key, so the green returns. Applying the overlay once is
    therefore not enough: it has to be held.

    Only touched when something is actually wrong, because SetWindowPos with
    SWP_FRAMECHANGED on every tick makes the window flicker.
    """
    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
    exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    want_style = style & ~_DECORATION
    want_ex = exstyle | WS_EX_LAYERED
    want_ex = (want_ex | WS_EX_TRANSPARENT) if click_through \
        else (want_ex & ~WS_EX_TRANSPARENT)

    changed = force or style != want_style or exstyle != want_ex
    if changed:
        user32.SetWindowLongW(hwnd, GWL_STYLE, want_style)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, want_ex)
        # Re-adding WS_EX_LAYERED resets the colour key, so it goes back too.
        user32.SetLayeredWindowAttributes(hwnd, key, 255, LWA_COLORKEY)
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                            _REFRESH | SWP_NOACTIVATE)
        return True
    return ensure_topmost(hwnd)


def restore(hwnd) -> None:
    saved = state()
    if saved.get("style") is not None:
        user32.SetWindowLongW(hwnd, GWL_STYLE, saved["style"])
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, saved["exstyle"])
    else:
        # No record of what it looked like. Clearing the transparency is not
        # enough on its own: without the decorations put back the window has no
        # title bar and no resize border, so it cannot be moved or resized by
        # hand -- which is most of what "restore" is asked for.
        print("no saved state; restoring plain defaults")
        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        user32.SetWindowLongW(hwnd, GWL_STYLE, style | _DECORATION)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                              user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                              & ~(WS_EX_LAYERED | WS_EX_TRANSPARENT))
    if STATE.exists():
        STATE.unlink()
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, _REFRESH)


# -- staying on top --------------------------------------------------------
def is_topmost(hwnd) -> bool:
    return bool(user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)


def ensure_topmost(hwnd) -> bool:
    """Put the window back on top if something demoted it.

    Setting HWND_TOPMOST once is not enough. VTube Studio is a Unity
    application and re-asserts its own window state, so the flag is silently
    dropped the first time another app takes the foreground -- which looks,
    from the outside, exactly like "always on top does not work".

    SWP_NOACTIVATE matters here: without it, every correction would yank focus
    away from whatever is being typed into.

    Returns True if it actually had to fix something.
    """
    if is_topmost(hwnd):
        return False
    if not user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                               SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE):
        raise ctypes.WinError(ctypes.get_last_error())
    return True


def start_keeper(interval: float = 1.0) -> threading.Event:
    """Hold the whole overlay in place in the background. Returns its stop event.

    Cheap enough to run continuously: two GetWindowLongW per tick, and a write
    only on the ticks where VTube Studio has actually undone something.
    """
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval):
            try:
                saved = state()
                if "key" not in saved:
                    continue          # overlay not applied; nothing to hold
                hwnd = find_window()
                if hwnd is not None:
                    ensure_overlay(hwnd, saved["key"],
                                   saved.get("click_through", False))
            except Exception:
                pass          # VTS closing mid-call must not kill the thread

    threading.Thread(target=loop, daemon=True).start()
    return stop


# -- click-through ---------------------------------------------------------
def is_click_through(hwnd) -> bool:
    return bool(user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TRANSPARENT)


def set_click_through(hwnd, on: bool) -> None:
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ex = (ex | WS_EX_TRANSPARENT) if on else (ex & ~WS_EX_TRANSPARENT)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex)
    user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, _REFRESH)


def toggle_click_through() -> bool | None:
    """Flip whether clicks land on her or on what is behind her.

    Returns the new state, or None if the VTube Studio window is not there.
    Deliberately independent of whether the overlay has been applied: clicks
    passing through an ordinary window is odd but harmless, and refusing to
    toggle would be the more confusing behaviour when the state file has been
    lost.
    """
    hwnd = find_window()
    if hwnd is None:
        return None
    wanted = not is_click_through(hwnd)
    set_click_through(hwnd, wanted)
    # Recorded, or the keeper would read the old preference off disk and undo
    # this within the second.
    saved = state()
    if saved:
        saved["click_through"] = wanted
        STATE.write_text(json.dumps(saved), encoding="utf-8")
    return wanted
