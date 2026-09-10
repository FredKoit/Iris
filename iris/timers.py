"""Timers, because it is the thing people actually say out loud to a voice.

"Remind me in ten minutes" is not conversation and must not be answered like
one: routing it through the brain gets a model agreeing to do something it has
no way of doing, which is worse than refusing. So it is parsed here and acted
on directly, the same shape as the memory commands in `commands.py`.

Everything above `Timers` is a pure function of a string, so the whole grammar
can be tested without a microphone, a clock or a model.

Two deliberate limits. Timers live in memory only and do not survive a restart
-- persisting a ten-minute countdown across a process that was not running for
most of it is a worse answer than losing it. And what she says when one fires
is a fixed line rather than a generated one, for the reason the memory
acknowledgements are fixed: a timer is a utility, one that waffles for two
sentences before saying what it was for is worse than one that does not, and a
generation that fails at the moment the timer fires is the single failure this
feature cannot afford.
"""

import random
import re
import threading
import time
from dataclasses import dataclass

from .commands import POLITE, TRAILING

SET = "set"
CANCEL = "cancel"
LEFT = "left"

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
# Whisper writes most numbers as digits, but not these: nobody's "a couple of
# minutes" comes back as "2 minutes".
_VAGUE = {"a": 1, "an": 1, "half": 0.5, "quarter": 0.25, "couple": 2,
          "few": 3, "several": 4}
_COUNTS = {**_ONES, **_TEENS, **_TENS, **_VAGUE}

_UNITS = {"second": 1, "seconds": 1, "sec": 1, "secs": 1,
          "minute": 60, "minutes": 60, "min": 60, "mins": 60,
          "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600}


def _alt(words) -> str:
    """Longest first, so "seconds" is not matched as "second" with a stray s."""
    return "|".join(sorted(words, key=len, reverse=True))


# One "<count> <unit>" pair. The leading article and the inner one are both
# optional and both needed: "a couple of minutes" puts it before the count and
# "half an hour" puts it after, and the regex backtracks between the two
# readings on its own.
_QUANTITY = re.compile(
    r"(?:\ban?\s+)?"
    r"\b(?P<count>\d+(?:\.\d+)?"
    rf"|(?:{_alt(_TENS)})[\s-]+(?:{_alt(_ONES)})"
    rf"|{_alt(_COUNTS)})\b"
    r"\s*(?:of\s+)?(?:an?\s+)?"
    rf"(?P<unit>{_alt(_UNITS)})\b",
    re.I,
)


# Digits, or a number said in words, with no unit attached. Shared with
# actions.py the way POLITE and TRAILING are shared out of commands.py: "set
# the volume to fifty" has to mean the same fifty "fifty minutes" does, and a
# second copy of these tables would drift from this one.
#
# The vague counts are deliberately not in it. "A", "half" and "a few" are
# quantities of minutes; "turn it up a bit" is not a request for one percent.
PLAIN = (r"\d+(?:\.\d+)?"
         rf"|(?:{_alt(_TENS)})[\s-]+(?:{_alt(_ONES)})"
         rf"|{_alt({**_ONES, **_TEENS, **_TENS})}")


def count(text: str) -> float:
    """The number a matched span of digits or words is worth.

    Public because actions.py parses percentages with the same words.
    """
    text = text.strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    if text in _COUNTS:
        return _COUNTS[text]
    # "twenty five" and "twenty-five" are one number in two words.
    return sum(_COUNTS.get(part, 0) for part in re.split(r"[\s-]+", text))


_FRACTIONS = {"half": 0.5, "halves": 0.5, "quarter": 0.25, "quarters": 0.25,
              "third": 1 / 3, "thirds": 1 / 3}

# "three quarters of an hour" -- a plural fraction with a count of its own,
# which the quantity pattern cannot read because "three quarters" sums to 3.25
# when taken word by word.
_FRACTION = re.compile(
    rf"\b(?P<n>\d+|{_alt(_ONES)})\s+(?P<frac>quarters|halves|thirds)\s+"
    rf"(?:of\s+)?(?:an?\s+)?(?P<unit>{_alt(_UNITS)})\b", re.I,
)

# "an hour and a half" -- a fraction with no unit at all, which means the unit
# in front of it.
_TRAILING_FRACTION = re.compile(r"\band\s+an?\s+(?P<frac>half|quarter|third)\b",
                                re.I)


def duration(text: str) -> float | None:
    """Seconds named in `text`, or None if it names none.

    Every quantity is summed, so "one hour thirty minutes" needs no pattern of
    its own. Returning None is what stops "set a timer" and "timer for the
    pasta" being treated as commands at all -- with no duration there is
    nothing to set, and letting them through would mean holding a
    half-finished command open across turns, which is a conversation this
    cannot have.
    """
    total = 0.0
    found = False

    # Fractions first, and the words they used are blanked out afterwards: left
    # in place, "three quarters of an hour" is also read as a whole "an hour".
    for m in _FRACTION.finditer(text):
        total += (count(m.group("n")) * _FRACTIONS[m.group("frac").lower()]
                  * _UNITS[m.group("unit").lower()])
        found = True
    text = _FRACTION.sub(lambda m: " " * len(m.group(0)), text)

    unit = None
    for m in _QUANTITY.finditer(text):
        unit = _UNITS[m.group("unit").lower()]
        total += count(m.group("count")) * unit
        found = True
    if unit is not None:
        tail = _TRAILING_FRACTION.search(text)
        if tail:
            total += _FRACTIONS[tail.group("frac").lower()] * unit

    return total if found and total > 0 else None


# Each is tried in turn and the first whose duration actually parses wins, so a
# pattern that matches the words but not the time falls through to the next
# rather than swallowing the line.
_SET_FORMS = (
    # "remind me to take the pizza out in ten minutes". The label is greedy on
    # purpose: it has to reach the *last* "in", or "put it in the oven in ten
    # minutes" is filed under "put it".
    re.compile(r"^\s*" + POLITE + r"remind\s+me\s+to\s+(?P<label>.+)"
               r"\s+in\s+(?P<dur>.+?)" + TRAILING, re.I),
    # "remind me in ten minutes to take the pizza out", label optional.
    re.compile(r"^\s*" + POLITE + r"remind\s+me\s+(?:in|after)\s+(?P<dur>.+?)"
               r"(?:\s+(?:to|about|that)\s+(?P<label>.+?))?" + TRAILING, re.I),
    # "set a timer for ten minutes", "timer for ten minutes".
    re.compile(r"^\s*" + POLITE + r"(?:set|start|put\s+on|make)?\s*(?:an?\s+)?"
               r"timers?\s+(?:for|on)?\s*(?P<dur>.+?)" + TRAILING, re.I),
    # "start a ten minute timer".
    re.compile(r"^\s*" + POLITE + r"(?:set|start|make)\s+(?:an?\s+)?"
               r"(?P<dur>.+?)\s+timer" + TRAILING, re.I),
    # "wake me in an hour".
    re.compile(r"^\s*" + POLITE + r"(?:wake|nudge|ping|poke|shout\s+at)\s+me"
               r"\s+(?:in|after)\s+(?P<dur>.+?)" + TRAILING, re.I),
)

_CANCEL = re.compile(
    r"^\s*" + POLITE + r"(?:cancel|stop|clear|kill|scrap|drop|forget)\s+"
    r"(?:the\s+|my\s+|that\s+|all\s+(?:the\s+|my\s+)?)?"
    r"(?:timers?|reminders?|alarms?)\b.*$", re.I,
)

# Two lookaheads rather than one pattern, so the words can arrive in any order:
# "how long left on that timer" and "the timer, how long".
_LEFT = re.compile(
    r"^(?=.*\b(?:timers?|reminders?|alarms?)\b)"
    r"(?=.*(?:how\s+long|how\s+much|left|remaining|to\s+go))", re.I,
)


def parse(text: str) -> tuple[str, float, str] | None:
    """Recognise a timer command. Returns (action, seconds, label), or None.

    `seconds` and `label` are meaningful only for SET.

    Cancelling is checked first, and this whole function has to run before
    `commands.parse` -- "forget the timer" is a cancellation, and the memory
    grammar reads it as an instruction to start deleting facts about timers.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if _CANCEL.match(stripped):
        return CANCEL, 0.0, ""
    if _LEFT.match(stripped):
        return LEFT, 0.0, ""
    for form in _SET_FORMS:
        m = form.match(stripped)
        if not m:
            continue
        seconds = duration(m.group("dur"))
        if seconds is None:
            continue
        label = (m.groupdict().get("label") or "").strip(" ,.")
        return SET, seconds, label
    return None


def spoken(seconds: float, article: bool = True) -> str:
    """A span said exactly, unlike `awareness.humanize`, which rounds.

    A timer set for an hour is an hour, not "about an hour". Rounding is right
    for "how long have we been talking" and wrong for the one thing here whose
    whole point is that it is precise.

    `article` is off for the possessive lines below. A single hour is naturally
    "an hour", and "your an hour is up" is not a sentence.
    """
    total = int(round(seconds))
    if total < 60:
        return "one second" if total == 1 else f"{total} seconds"
    one_hour, one_minute = ("an hour", "a minute") if article else ("hour", "minute")
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(one_hour if hours == 1 else f"{hours} hours")
    if minutes:
        parts.append(one_minute if minutes == 1 else f"{minutes} minutes")
    if secs and not hours:
        parts.append("one second" if secs == 1 else f"{secs} seconds")
    return " and ".join(parts)


def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


# {Span} is the capitalised form and {bare} the one with no leading article.
# Every span here opens a sentence, so a span that begins "an hour" needs the
# capital, and the two possessive lines need it gone entirely.
_SET_LINES = ("Fine. {Span}.", "{Span}.", "Right. {Span}.", "{Span}. Noted.")
_FIRE_LINES = ("That's your {bare}.", "{Span}, up.", "Timer. {Span}.",
               "Your {bare} is up.")


@dataclass
class Timer:
    span: float        # what was asked for
    label: str
    at: float          # monotonic deadline


class Timers:
    """The ones outstanding, and what she says about them.

    Deadlines are monotonic, not wall clock: a ten-minute timer is ten minutes
    even if the system clock is corrected underneath it.
    """

    def __init__(self, cfg, seed: int | None = None):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._pending: list[Timer] = []
        self._rng = random.Random(seed)

    def add(self, span: float, label: str = "") -> Timer:
        timer = Timer(span=span, label=label, at=time.monotonic() + span)
        with self._lock:
            self._pending.append(timer)
            self._pending.sort(key=lambda t: t.at)
        return timer

    def due(self) -> list[Timer]:
        """Everything whose time is up, removed from the list."""
        now = time.monotonic()
        with self._lock:
            fired = [t for t in self._pending if t.at <= now]
            self._pending = [t for t in self._pending if t.at > now]
        return fired

    def pending(self) -> list[Timer]:
        with self._lock:
            return list(self._pending)

    def cancel_all(self) -> int:
        with self._lock:
            n = len(self._pending)
            self._pending.clear()
        return n

    # -- what she says -----------------------------------------------------
    def _fill(self, template: str, span: float) -> str:
        return template.format(Span=_cap(spoken(span)),
                               bare=spoken(span, article=False))

    def acknowledge(self, span: float, label: str = "") -> str:
        line = self._fill(self._rng.choice(_SET_LINES), span)
        return f"{line} {_cap(label)}." if label else line

    def announce(self, fired: list[Timer]) -> str:
        """One line for however many went off at once."""
        if not fired:
            return ""
        first = fired[0]
        line = self._fill(self._rng.choice(_FIRE_LINES), first.span)
        if first.label:
            line += f" {_cap(first.label)}."
        # Two at once is rare enough not to deserve a sentence each, and she
        # does not do sentences each.
        if len(fired) > 1:
            line += f" And {len(fired) - 1} more."
        return line

    def remaining_line(self) -> str:
        outstanding = self.pending()
        if not outstanding:
            return "Nothing running."
        left = max(0.0, outstanding[0].at - time.monotonic())
        line = f"{_cap(spoken(left))} left."
        if len(outstanding) > 1:
            line += f" {len(outstanding) - 1} more after it."
        return line
