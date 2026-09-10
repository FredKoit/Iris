"""Reaching the desktop she is already floating on.

`screen.py` gave her eyes on this machine; this gives her hands. It is a much
shorter reach than that sounds -- four things people actually say out loud to
something sitting on their desktop: the volume, the brightness, whatever is
playing, and "open Spotify".

Deterministic on purpose, and that is the whole design decision. The assistant
this was ported from sends anything ambiguous to a second model call that
returns a JSON plan, and that is the one part of it that could not come across.
Iris runs a 2048-token context at temperature 0.9, tuned for a personality that
wanders -- the worst setting there is for emitting valid JSON. A tool guide
would not fit beside the persona and the memories at that size, and a second
generation in the middle of a turn is exactly the cost `timers.py` refuses to
pay merely to say "fine, ten minutes". So the grammar is regex, the same shape
as the timer and memory grammars, and anything it does not recognise falls
through to the brain untouched. That is the right answer for "can you turn my
life around", and it is not a limitation.

Two halves, split the way `screen.py` splits them, so only the second needs
Windows:

- Everything above `Desk` is a pure function of its arguments and can be tested
  with no speakers, no monitor and nothing installed to point at. The two
  things it would otherwise have to reach for -- what "it" last referred to,
  and which applications exist -- are passed in for that reason.
- `Desk` is the half that touches pycaw, WMI and user32. Unlike `screen.py`,
  the platform calls are looked up inside it rather than at module scope,
  because the grammar above has to import on a machine that has none of them.

It deliberately recognises verbs it may not be able to perform. With pycaw
missing, "turn it down" still parses and `Desk` says in one line that it
cannot -- because the alternative is handing that sentence to a model which
will cheerfully agree that it has. A companion that claims to have done
something is worse than one that admits it cannot, which is `timers.py`'s
argument about promises, unchanged.
"""

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from .commands import POLITE, TRAILING
from .timers import PLAIN, count

# What is being touched.
VOLUME = "volume"
BRIGHTNESS = "brightness"
MEDIA = "media"
APP = "app"

# What is being done to it.
SET = "set"
UP = "up"
DOWN = "down"
QUERY = "query"
MUTE = "mute"
UNMUTE = "unmute"
PLAY = "play"
NEXT = "next"
PREV = "previous"
STOP = "stop"
LAUNCH = "launch"


@dataclass(frozen=True)
class Act:
    """One recognised instruction.

    `value` is a percentage for the knobs, an application name for a launch,
    and None everywhere else -- including a step whose size was not named,
    which means one step of `action_step`.
    """
    kind: str
    op: str
    value: float | str | None = None


# --- the grammar -----------------------------------------------------------
#
# Every pattern below is anchored and wrapped in the POLITE/TRAILING pair out
# of commands.py, so "okay Iris, could you turn the volume down please" is the
# same sentence as "volume down". That wrapper is why this file does not need
# its own set of a hundred literal phrasings.
#
# TRAILING absorbs a trailing "please", which is all the memory grammar ever
# needed. An instruction to *do* something also arrives with an urgency adverb
# on the end -- "mute it now", "open spotify now" -- and nobody ever says
# "remember that now", so this is the first grammar to meet one. Without it the
# lazy name group in the launch pattern swallowed the adverb and went looking
# for an application called "spotify now".
_END = r"(?:\s+(?:now|already|right\s+now))?" + TRAILING

# "Screen" and "display" mean light, not sound, which is why they are here and
# not in the volume list: "turn the screen down" is about brightness. Neither
# is allowed into the query patterns -- "what is on my screen" is a question
# for the brain and shares every word with a brightness check.
_BRIGHT = re.compile(r"\b(?:brightness|backlight|screen|display|monitor)\b", re.I)
_VOL = re.compile(r"\b(?:volume|sound|audio|speakers?)\b", re.I)
_IT = re.compile(r"\b(?:it|that|this)\b", re.I)

# Adjectives that name the knob without naming the noun. "Make it louder" has
# no more said the word "volume" than "dim it" has said "brightness", and both
# are how people actually ask.
_MEANS_VOL = re.compile(r"\b(?:louder|quieter|softer)\b", re.I)
_MEANS_BRIGHT = re.compile(r"\b(?:brighter|dimmer|darker|brighten|dim|darken)\b", re.I)

_UP = re.compile(
    r"\b(?:up|raise|increase|boost|crank|louder|higher|brighten|brighter)\b", re.I)
_DOWN = re.compile(
    r"\b(?:down|lower|reduce|decrease|drop|quiet|quieter|softer"
    r"|dim|dimmer|darken|darker)\b", re.I)

# The gate that separates an instruction from a remark. Without it "the screen
# is at fifty percent" reads as an order to set the brightness to fifty, and
# "the volume is at maximum" as an order to raise it -- both are things
# somebody says *about* a machine, and both share every content word with the
# command. Requiring the sentence to open like an instruction is what tells
# them apart, and it is the same trick the memory grammar uses to leave "I
# remember going there" alone.
_INSTRUCTION = re.compile(
    r"^\s*" + POLITE +
    r"(?:(?:set|put|change|make|turn|bring|adjust|crank|raise|lower|increase"
    r"|decrease|reduce|boost|dim|darken|brighten|drop)\b"
    r"|(?:louder|quieter|softer|brighter|dimmer|darker)\b"
    r"|(?:max|maximum|full|half|halfway|minimum|zero)\s+"
    r"(?:volume|sound|audio|brightness|screen|display)\b"
    r"|(?:volume|sound|audio|brightness|screen|display)\s+(?:to|at|up|down)\b)",
    re.I,
)

# "to sixty", "at about sixty". PLAIN comes from timers.py so that the fifty in
# "set the volume to fifty" is the same fifty as in "fifty minutes".
_TO = re.compile(
    rf"\b(?:to|at)\s+(?:about\s+|around\s+|roughly\s+)?(?P<n>{PLAIN})\s*(?:%|percent)?\b",
    re.I)
_BY = re.compile(rf"\bby\s+(?P<n>{PLAIN})\b", re.I)
# A number wearing a unit. Kept apart from _ANY because this one is safe to
# read as an amount even when no direction word is present.
_PCT = re.compile(rf"\b(?P<n>{PLAIN})\s*(?:%|percent)\b", re.I)
_ANY = re.compile(rf"\b(?P<n>{PLAIN})\b", re.I)

_FULL = re.compile(r"\b(?:max|maximum|full|full\s+blast|all\s+the\s+way\s+up)\b", re.I)
# "min" is not in here on purpose: it is a minute far more often than it is a
# minimum, and timers.py is entitled to it.
_ZERO = re.compile(
    r"\b(?:minimum|zero|nothing|silent|off|all\s+the\s+way\s+down)\b", re.I)
_HALF = re.compile(r"\bhalf(?:way)?\b", re.I)

_VOL_Q = re.compile(
    r"^\s*" + POLITE + r"(?:what(?:'s| is)|how)\b[^?]*\b(?:volume|loud)\b", re.I)
_BRIGHT_Q = re.compile(
    r"^\s*" + POLITE + r"(?:what(?:'s| is)|how)\b[^?]*\bbright(?:ness)?\b", re.I)

# Bare "mute" is a command. It was not, at first, on the reasoning that said to
# something which is listening to you it might mean her microphone -- and that
# was wrong twice over. There is no voice path to her microphone at all: it is
# a tray switch and a hotkey, so there was never a second reading to be
# ambiguous with. And muting her own ears by voice is not a thing anybody wants
# anyway, because it cannot be undone the same way.
#
# What withholding it actually bought was the brain answering instead, which it
# did with total confidence and no idea: "Mute is already active", then "Volume
# is at 57. Muting would bring it to zero", then "Mute command issued. Final
# state: zero. Done." Nothing was muted at any point. A grammar that declines a
# real instruction does not leave a polite silence for someone else to fill --
# it hands the sentence to the one component in this process that will make up
# an answer, which is the failure this whole file exists to avoid.
_OBJECT = (r"(?:the\s+)?(?:sound|audio|volume|music|speakers?|system|computer"
           r"|everything|it|that|this)")
_MUTE = re.compile(
    r"^\s*" + POLITE + r"(?:mute|silence)(?:\s+" + _OBJECT + r")?" + _END, re.I)
_UNMUTE = re.compile(
    r"^\s*" + POLITE + r"(?:un-?mute(?:\s+" + _OBJECT + r")?"
    r"|turn\s+(?:the\s+)?(?:sound|audio|volume)\s+back\s+on)" + _END, re.I)
# The one reading that is genuinely not this. She cannot mute her own
# microphone on command, and muting the speakers when asked to would be a
# confident wrong answer of exactly the kind described above.
_HER_EARS = re.compile(r"\b(?:mic|microphone|yourself|ears)\b", re.I)

# Bare "stop" is not here either: she is a thing you interrupt, and "stop" said
# out loud while she is talking means stop talking. Stopping the music has to
# say so.
_THING = r"(?:the\s+)?(?:music|song|track|audio|playback|video|it|this)"
_STOP = re.compile(
    r"^\s*" + POLITE + r"stop\s+(?:the\s+)?(?:music|song|track|playback|playing)"
    + _END, re.I)
_PLAY = re.compile(
    r"^\s*" + POLITE + r"(?:play|resume|unpause|pause)(?:\s+" + _THING + r")?"
    + _END, re.I)
_NEXT = re.compile(
    r"^\s*" + POLITE + r"(?:next|skip)(?:\s+(?:this|the)?\s*(?:song|track|one))?"
    + _END, re.I)
# "Last" and "previous" need the noun. Bare "last" is a word in a sentence far
# more often than it is a request for the previous track.
_PREV = re.compile(
    r"^\s*" + POLITE + r"(?:(?:previous|last|prior)\s+(?:song|track)"
    r"|(?:go\s+)?back\s+a\s+(?:song|track))" + _END, re.I)

_LAUNCH = re.compile(
    r"^\s*" + POLITE + r"(?:open|launch|start|run|fire\s+up|bring\s+up|pull\s+up)"
    r"\s+(?P<what>.+?)(?:\s+for\s+me)?" + _END, re.I)
# Only the pronoun-resolving half of a router may turn these into a name, and
# this file does not have one. Launching an application literally called "it"
# is not a thing anybody has ever wanted.
_NOT_A_NAME = frozenset("it that this one up them something anything".split())


def norm(name: str) -> str:
    """The form an application name is stored and looked up under."""
    return re.sub(r"\s+", " ", name.strip().strip("\"'")).lower()


def compact(name: str) -> str:
    """...and the form that survives punctuation and a mishearing of it."""
    return re.sub(r"[^a-z0-9 ]", "", norm(name)).strip()


def _pct(value: float) -> float:
    return max(0.0, min(100.0, value))


def _media(t: str) -> Act | None:
    if _STOP.match(t):
        return Act(MEDIA, STOP)
    if _NEXT.match(t):
        return Act(MEDIA, NEXT)
    if _PREV.match(t):
        return Act(MEDIA, PREV)
    if _PLAY.match(t):
        return Act(MEDIA, PLAY)
    return None


def _mute(t: str) -> Act | None:
    if _HER_EARS.search(t):
        return None
    if _MUTE.match(t):
        return Act(VOLUME, MUTE)
    if _UNMUTE.match(t):
        return Act(VOLUME, UNMUTE)
    return None


def _knob(t: str, last: str) -> Act | None:
    """The volume and the brightness, which are the same sentence twice.

    `last` is whichever of them was touched most recently, and it is the only
    thing that can make sense of "turn it down". The caller owns it, and owns
    deciding when it has gone stale -- an hour later "it" is not the volume any
    more, and a module that cannot see a clock should not be the one guessing.
    """
    if _BRIGHT_Q.match(t):
        return Act(BRIGHTNESS, QUERY)
    if _VOL_Q.match(t):
        return Act(VOLUME, QUERY)
    if not _INSTRUCTION.match(t):
        return None

    bright, vol = _BRIGHT.search(t), _VOL.search(t)
    if bright and vol:
        # Both named, so the one said first is the one meant: "turn the screen
        # down, not the sound".
        kind = BRIGHTNESS if bright.start() < vol.start() else VOLUME
    elif bright:
        kind = BRIGHTNESS
    elif vol:
        kind = VOLUME
    elif _MEANS_BRIGHT.search(t):
        kind = BRIGHTNESS
    elif _MEANS_VOL.search(t):
        kind = VOLUME
    elif last and _IT.search(t):
        kind = last
    else:
        return None

    # An explicit target beats everything, but only when it is a target and not
    # a step: "up by ten" has a number after a preposition too.
    to = _TO.search(t)
    if to and not _BY.search(t):
        return Act(kind, SET, _pct(count(to.group("n"))))
    if _FULL.search(t):
        return Act(kind, SET, 100.0)
    if _ZERO.search(t):
        return Act(kind, SET, 0.0)
    if _HALF.search(t):
        return Act(kind, SET, 50.0)

    up, down = _UP.search(t), _DOWN.search(t)
    if not (up or down):
        # "Make the volume fifty percent" names a target with no preposition
        # and no direction. The unit is what makes it safe to read as one.
        pct = _PCT.search(t)
        return Act(kind, SET, _pct(count(pct.group("n")))) if pct else None

    amount = _BY.search(t) or _PCT.search(t) or _ANY.search(t)
    size = _pct(count(amount.group("n"))) if amount else None
    if down and (not up or down.start() < up.start()):
        return Act(kind, DOWN, size)
    return Act(kind, UP, size)


def _launch(t: str, known) -> Act | None:
    """Open something, but only something already known to exist.

    An unrecognised name returns None and the sentence goes to the brain, which
    is both the friendlier answer to "open the pod bay doors" and the reason
    this is safe. Every word here arrived from a microphone by way of a
    transcriber that guesses; handing an unresolved guess to os.startfile or
    Popen would let a mishearing name a path or a binary. The index is the
    whitelist, and there is no fallback past it.
    """
    m = _LAUNCH.match(t)
    if not m:
        return None
    what = norm(m.group("what"))
    if not what or what in _NOT_A_NAME:
        return None
    if what in known:
        return Act(APP, LAUNCH, what)
    short = compact(what)
    if short and short in known:
        return Act(APP, LAUNCH, short)
    return None


def parse(text: str, last: str = "", known=frozenset()) -> Act | None:
    """Recognise an instruction to do something. Returns an Act, or None.

    None means this was not one, and the caller should answer it normally.
    """
    t = text.strip()
    if not t:
        return None
    return _media(t) or _mute(t) or _knob(t, last) or _launch(t, known)


# --- the machine ------------------------------------------------------------

# Names that resolve without anything being scanned, either because the
# executable is not what anybody calls it (winword, msedge) or because the
# thing is a URI protocol and has no file on disk to find at all.
KNOWN: dict[str, str] = {
    "notepad": "notepad",
    "calculator": "ms-calculator:",
    "settings": "ms-settings:",
    "paint": "mspaint",
    "explorer": "explorer",
    "file explorer": "explorer",
    "spotify": "spotify:",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "discord": "discord",
    "teams": "msteams:",
    "microsoft teams": "msteams:",
    "word": "winword",
    "microsoft word": "winword",
    "excel": "excel",
    "microsoft excel": "excel",
    "powershell": "powershell",
    "terminal": "wt.exe",
}

_SHORTCUTS = frozenset((".lnk", ".url", ".exe"))
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_CANNOT = {
    VOLUME: "I can't reach the volume.",
    BRIGHTNESS: "I can't reach the brightness.",
    MEDIA: "Nothing took that.",
    APP: "That didn't open.",
}


class Desk:
    """The parts of this machine she can actually reach.

    Every operation returns the line she should say and, sometimes, one thing
    to do *after* she has said it. That second half exists for exactly one case
    and is worth the shape it costs: muting the system silences her own voice,
    so muting before the acknowledgement plays means the acknowledgement is
    never heard. She says "muted", and then it is.

    Nothing here raises. A missing package, or a monitor that will not answer,
    becomes a sentence admitting it, with the real error printed to the
    terminal where it is of use to somebody.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Replaced wholesale by the indexing thread rather than mutated, so
        # names() never needs a lock and never sees a half-built table.
        self.apps: dict[str, str] = dict(KNOWN)
        self._endpoint = None
        threading.Thread(target=self._index, name="apps", daemon=True).start()

    def names(self) -> frozenset:
        """What `parse` is allowed to turn into a launch.

        Called mid-turn, so it never waits for the scan: until that lands only
        the curated names resolve, which is a slower start and not a failure.
        """
        return frozenset(self.apps)

    # -- the application index ---------------------------------------------
    def _index(self) -> None:
        """Every shortcut on the Start Menu and the desktop, by name.

        Off the turn loop entirely: a cold rglob over three trees runs into
        hundreds of milliseconds, and the first thing this process does is load
        three models. Nobody says "open Spotify" in the first second.
        """
        roots = [
            Path(os.environ.get("ProgramData", r"C:\ProgramData"))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path(os.environ.get("APPDATA", ""))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs",
            Path.home() / "Desktop",
        ]
        found: dict[str, str] = {}
        for root in roots:
            try:
                if not root.is_dir():
                    continue
                for path in root.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in _SHORTCUTS:
                        continue
                    name = norm(path.stem)
                    if not name:
                        continue
                    found.setdefault(name, str(path))
                    # "Visual Studio Code" is also asked for without the
                    # punctuation, which is all Whisper ever writes.
                    short = compact(name)
                    if short and short != name:
                        found.setdefault(short, str(path))
            except (OSError, PermissionError):
                continue
        # The curated table settles the rest, but not blindly, because its two
        # kinds of entry deserve opposite treatment. A protocol handler always
        # wins: "spotify:" reaches the installed application however it was
        # installed, where a Start Menu shortcut of that name may well point at
        # a stale updater. A bare executable name is the other way round --
        # "discord" resolves only if it happens to be on PATH, and a shortcut
        # actually found on this machine beats a guess about this machine.
        for name, target in KNOWN.items():
            if ":" in target or name not in found:
                found[name] = target
        self.apps = found

    # -- doing it ------------------------------------------------------------
    def do(self, act: Act):
        """Carry out one Act. Returns (line, after), where `after` may be None."""
        try:
            if act.kind == VOLUME:
                return self._do_volume(act)
            if act.kind == BRIGHTNESS:
                return self._do_brightness(act)
            if act.kind == MEDIA:
                return self._do_media(act)
            return self._do_launch(act)
        except Exception as exc:
            print(f"  [{act.kind} {act.op} failed: {type(exc).__name__}: {exc}]",
                  flush=True)
            return _CANNOT.get(act.kind, "That didn't work."), None

    def _step(self, act: Act) -> float:
        return self.cfg.action_step if act.value is None else float(act.value)

    # -- volume --------------------------------------------------------------
    def _speakers(self):
        """The master volume endpoint, resolved once and kept.

        Imported here rather than at the top of the file: pycaw and comtypes
        are not needed to hold a conversation, and this module has to import
        without them so that the grammar above can be tested anywhere.

        Two pycaw shapes, because the one installed here is not the one every
        example on the internet describes. Older releases return a raw COM
        device you have to Activate yourself; current ones return an
        AudioDevice that has already done it and exposes the result as
        .EndpointVolume -- on which Activate does not exist at all, which is a
        clean AttributeError and no hint as to why. Taking the finished
        interface when there is one, and activating a device when there is not,
        covers both.
        """
        if self._endpoint is None:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            speakers = AudioUtilities.GetSpeakers()
            ready = getattr(speakers, "EndpointVolume", None)
            if ready is not None and hasattr(ready, "GetMasterVolumeLevelScalar"):
                self._endpoint = ready
            else:
                itf = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._endpoint = cast(itf, POINTER(IAudioEndpointVolume))
        return self._endpoint

    def _do_volume(self, act: Act):
        vol = self._speakers()
        now = int(round(vol.GetMasterVolumeLevelScalar() * 100))
        muted = bool(vol.GetMute())

        if act.op == QUERY:
            return (f"Muted. It was on {now}." if muted else f"{now}."), None
        if act.op == MUTE:
            return "Muted.", lambda: vol.SetMute(1, None)
        if act.op == UNMUTE:
            vol.SetMute(0, None)
            return "Back on.", None

        if act.op == SET:
            target = float(act.value)
        elif act.op == UP:
            target = now + self._step(act)
        else:
            target = now - self._step(act)
        target = int(max(0, min(100, round(target))))

        def apply() -> None:
            vol.SetMasterVolumeLevelScalar(target / 100.0, None)
            # Raising the volume of something muted does nothing audible, and
            # "turn it up" while muted plainly means turn it back on.
            if muted and target > 0:
                vol.SetMute(0, None)

        # Zero is mute by another name and has the same problem: applied first,
        # the number announcing it is never heard.
        if target == 0:
            return "Off.", apply
        apply()
        return f"{target}.", None

    # -- brightness ----------------------------------------------------------
    # Reading the panel. There is no ctypes call for brightness the way there
    # is for volume; it is WMI, and WMI means a PowerShell.
    _READ = ("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness"
             " | Select-Object -First 1 -ExpandProperty CurrentBrightness)")

    def _powershell(self, script: str) -> str:
        """One PowerShell, bounded, silent, and made to fail loudly.

        The preamble is not decoration. A failed WMI method call is a
        *non-terminating* error by default: PowerShell prints it to stderr and
        then exits 0, so a caller checking the return code sees success. The
        first version of this file did exactly that -- it said "eighty" in a
        pleasant voice while changing nothing whatsoever, which is precisely
        the failure the module docstring claims not to have. `Stop` turns that
        into an exit code, and do() turns an exit code into an admission.

        Bounded by a timeout so a monitor that never answers costs one turn
        rather than the session, and launched with CREATE_NO_WINDOW so a
        console does not flash over the game she is floating on.
        """
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference='Stop';" + script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=self.cfg.action_timeout_s, creationflags=_NO_WINDOW,
        )
        if out.returncode != 0:
            detail = " ".join(out.stderr.split())[:200]
            raise RuntimeError(detail or "the monitor refused it")
        return out.stdout.strip()

    def _do_brightness(self, act: Act):
        if act.op == QUERY:
            return f"{int(self._powershell(self._READ).splitlines()[0])}.", None

        if act.op == SET:
            want = str(int(max(0, min(100, round(float(act.value))))))
        else:
            delta = int(round(self._step(act))) * (1 if act.op == UP else -1)
            # Read, add and write inside the one process. Doing it as two
            # subprocesses is two PowerShell starts, and a PowerShell start is
            # 0.85s on this machine -- longer than she takes to begin
            # answering an actual question.
            want = f"[Math]::Max(0,[Math]::Min(100,{self._READ} + ({delta})))"

        # WmiSetBrightness is an *instance* method. `Invoke-CimMethod
        # -ClassName` is the spelling that looks right and calls a static one,
        # and the error it gives back ("Invalid method Parameter(s)") says
        # nothing at all about which half of that is wrong.
        got = self._powershell(
            f"$t = {want};"
            " $m = Get-CimInstance -Namespace root/WMI"
            " -ClassName WmiMonitorBrightnessMethods | Select-Object -First 1;"
            " Invoke-CimMethod -InputObject $m -MethodName WmiSetBrightness"
            " -Arguments @{Timeout=[uint32]1;Brightness=[byte]$t} | Out-Null;"
            " Write-Output $t"
        )
        return f"{int(got.splitlines()[-1])}.", None

    # -- whatever is playing -------------------------------------------------
    # The media keys a keyboard with buttons on it would send. Whichever
    # application owns the system media session receives them, so she never has
    # to know whether it is Spotify or a browser tab.
    _KEYS = {PLAY: 0xB3, NEXT: 0xB0, PREV: 0xB1, STOP: 0xB2}
    _SAID = {PLAY: "Done.", NEXT: "Next.", PREV: "Back.", STOP: "Stopped."}

    def _do_media(self, act: Act):
        import ctypes

        user32 = ctypes.windll.user32
        vk = self._KEYS[act.op]
        user32.keybd_event(vk, 0, 0x0001, 0)              # EXTENDEDKEY
        user32.keybd_event(vk, 0, 0x0001 | 0x0002, 0)     # ... | KEYUP
        return self._SAID[act.op], None

    # -- opening things ------------------------------------------------------
    def _do_launch(self, act: Act):
        target = self.apps.get(str(act.value))
        if target is None:
            # Only reachable if the index was rebuilt between parse and here.
            return f"I don't have {act.value}.", None
        if Path(target).exists():
            os.startfile(target)
        elif ":" in target:
            os.startfile(target)          # a protocol handler, not a path
        else:
            subprocess.Popen([target], shell=False)
        return f"Opening {act.value}.", None
