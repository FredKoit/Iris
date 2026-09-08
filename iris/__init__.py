import sys

from .config import Config

__all__ = ["Config"]


def _readable_console() -> None:
    """Stop a non-ASCII character from killing the process on Windows.

    Python picks the console's own code page for stdout, which here is cp1252.
    Print a Japanese expression name, a Chinese hotkey name, or anything the
    model emits outside that page and `print` raises UnicodeEncodeError --
    ending a conversation, or a setup script, over a character that was only
    ever going to be looked at. `scripts/avatar.py` died exactly this way while
    listing an unmapped hotkey.

    UTF-8 so the names are actually legible, and errors="replace" so that even
    on a console that cannot render them the worst case is a "?" instead of a
    traceback.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass          # redirected to something that cannot be reconfigured


_readable_console()
