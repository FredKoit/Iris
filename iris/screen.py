"""What is in front of her.

She is pinned above whatever else is on this desktop and, until now, had no idea
what that was. `awareness.py` exists because a companion with no clock is
unmoored; a companion floating over a game that cannot tell a game from a
spreadsheet is unmoored in the same way and for the same reason. One string per
turn fixes it, and it is the difference between an assistant that answers
questions and something that lives here.

Two halves, split so only the first is Windows-only:

- `foreground()` is the platform call, and it is all user32/kernel32.
- `Watcher` does the accounting, and `observe()` takes a reading as arguments
  rather than fetching one, so the timekeeping can be tested on a machine with
  no windows at all.

None of this leaves the laptop. Ollama is on 127.0.0.1 and the prompt is never
written to disk -- but window titles are the most revealing text on a machine,
so `screen_private` exists to keep named applications down to their name, and
the whole thing is off with `screen = False`.
"""

import ctypes
import ctypes.wintypes as wt
import threading
import time

from .config import Config

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Enough to ask a process for its own path, and no more. The full
# PROCESS_QUERY_INFORMATION is refused for anything running elevated, which
# would have meant losing the name of exactly the applications most worth
# knowing about.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Declared for the same reason as the ones in overlay.py: undeclared, ctypes
# passes and returns handles as C ints, and a 64-bit HWND or HANDLE comes back
# truncated.
user32.GetForegroundWindow.restype = wt.HWND
user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
kernel32.OpenProcess.restype = wt.HANDLE
kernel32.QueryFullProcessImageNameW.argtypes = [
    wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)
]
kernel32.QueryFullProcessImageNameW.restype = wt.BOOL
kernel32.CloseHandle.argtypes = [wt.HANDLE]


def _process_name(hwnd) -> str:
    """The executable behind a window, without its path or its .exe.

    Identity comes from the process and not from the title, because a title
    changes every time you switch file or browser tab. Tracking on it would
    restart the clock a hundred times an hour and never once notice that
    somebody has been in the same editor all evening.
    """
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value
    )
    if not handle:
        return ""              # a process this one is not allowed to ask about
    try:
        size = wt.DWORD(512)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size)
        ):
            return ""
        return buf.value.rsplit("\\", 1)[-1].removesuffix(".exe")
    finally:
        kernel32.CloseHandle(handle)


def foreground() -> tuple[str, str] | None:
    """(application, window title) for whatever has focus, or None.

    None covers the cases where the question has no answer: the lock screen,
    the moment between one window closing and the next taking over, a secure
    desktop. Those are not worth reporting and must not be mistaken for the
    user having switched to something.
    """
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    app = _process_name(hwnd)
    if not app:
        return None
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return app, buf.value.strip()


def _matches(name: str, patterns) -> bool:
    """Case-insensitive, and a substring counts, so "KeePass" catches
    "KeePassXC" and nobody has to know the exact executable name."""
    low = name.lower()
    return any(p.strip().lower() in low for p in patterns if p.strip())


class Watcher:
    """Polls the foreground window and totals how long each app has held it.

    The total, not the current run: people alt-tab constantly, and "you have
    had that game up for two hours" is the true and interesting statement,
    where "in front for forty seconds" is neither.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._app = ""
        self._title = ""
        self._held: dict[str, float] = {}    # app -> seconds in front, this session
        self._last: float | None = None      # when the last reading was taken

    # -- accounting --------------------------------------------------------
    def observe(self, app: str, title: str, now: float) -> None:
        """Fold one reading into the totals."""
        if (not app or _matches(app, self.cfg.screen_ignore)
                or (title and _matches(title, self.cfg.screen_ignore_titles))):
            # Her own window is not something they are doing, and clicking on
            # her must not make her forget what they were. Dropping `_last`
            # charges the gap to nobody rather than to whoever was in front
            # before it.
            #
            # Shell surfaces go the same way and have to be matched on the
            # title: the tray flyout and the desktop are both explorer, and so
            # is File Explorer, which is a real thing to be doing.
            self._last = None
            return
        with self._lock:
            if self._last is not None and self._app:
                # Credited to whoever held the front *during* the interval,
                # which is the previous reading's app, not this one's.
                self._held[self._app] = (
                    self._held.get(self._app, 0.0) + (now - self._last)
                )
            self._app = app
            self._title = title[:self.cfg.screen_title_max].strip()
            self._last = now

    def current(self) -> tuple[str, str, float] | None:
        """(app, title, seconds in front this session), or None if unknown."""
        with self._lock:
            if not self._app:
                return None
            title = "" if _matches(self._app, self.cfg.screen_private) \
                else self._title
            return self._app, title, self._held.get(self._app, 0.0)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.wait(self.cfg.screen_poll_s):
            try:
                seen = foreground()
                if seen is not None:
                    self.observe(seen[0], seen[1], time.time())
            except Exception:
                pass       # a window dying mid-call must not kill the thread

    def stop(self) -> None:
        self._stop.set()
