"""Where she is in time.

Without this she is genuinely unmoored: asked how long we had been talking she
answered "probably around last night's pizza dinner", and asked the day,
"Tuesday... probably Thursday". A language model has no clock, and a companion
that cannot tell morning from midnight cannot react to either.

Everything here is plain text destined for the system prompt, so it is written
the way it should be read back: qualitative first, exact second.
"""

import re
import time
from datetime import datetime

# Rough, but the point is that she has an opinion about the hour, not that the
# boundaries are correct.
_DAY_PARTS = (
    (5, "the middle of the night"),
    (8, "early morning"),
    (12, "the morning"),
    (14, "the middle of the day"),
    (18, "the afternoon"),
    (22, "the evening"),
    (24, "late at night"),
)


def part_of_day(hour: int) -> str:
    for limit, name in _DAY_PARTS:
        if hour < limit:
            return name
    return "late at night"


def humanize(seconds: float) -> str:
    """A span said the way a person would say it."""
    minutes = int(seconds // 60)
    if minutes < 1:
        return "less than a minute"
    if minutes == 1:
        return "a minute"
    if minutes < 45:
        return f"{minutes} minutes"
    hours = seconds / 3600
    if hours < 1.5:
        return "about an hour"
    if hours < 24:
        return f"about {round(hours)} hours"
    days = round(hours / 24)
    return "a day" if days == 1 else f"{days} days"


_SLUG = re.compile(r"[^a-z0-9]")


def _app_name(app: str, title: str) -> str:
    """The application said the way a person would say it.

    A process name is what the file on disk is called, not what the thing is:
    "eldenring", "Code", "chrome". Window titles conventionally end with the
    real name, so where a segment of the title contains the process name, that
    segment is the better label -- "Visual Studio Code" rather than "Code".
    """
    want = _SLUG.sub("", app.lower())
    if not want:
        return app
    for part in reversed(title.split(" - ")):
        part = part.strip()
        if part and want in _SLUG.sub("", part.lower()):
            return part
    return app


def doing(app: str, title: str = "", held: float = 0.0,
          settled: float = 300.0) -> str:
    """One line about what is in front of them, for the system prompt.

    Written as a label and not as a sentence, which is the whole trick. Given
    "They have had it up for about two hours so far today" a 3B model hands the
    clause straight back -- measured, it said "that's how long you've had it
    open today" out loud -- because a well-formed sentence in the prompt is
    ready to speak. A parenthesised label has to be reworded before it can be
    said at all, and rewording it is the point.

    The title leads and is quoted, marking it as a literal string rather than
    prose to carry on into. The application is named only when the title does
    not already say it: "ELDEN RING" needs no gloss, "motion.py - Iris" does.
    The duration appears only once it is long enough to be an observation
    rather than a stopwatch reading.
    """
    app, title = app.strip(), title.strip()
    if not app:
        return ""
    name = _app_name(app, title)
    note = []
    if title and _SLUG.sub("", name.lower()) not in _SLUG.sub("", title.lower()):
        note.append(name)
    if held >= settled:
        note.append(f"up {humanize(held)} today")
    what = f'"{title}"' if title else name
    return f"On screen in front of them: {what}" + (
        f" ({', '.join(note)})" if note else "")


def situation(started: float, last_seen: float | None = None,
              now: float | None = None,
              screen: tuple[str, str, float] | None = None,
              settled: float = 300.0) -> str:
    """The block describing this moment, for the system prompt."""
    now = time.time() if now is None else now
    stamp = datetime.fromtimestamp(now)

    lines = ["Right now:"] + [
        f"It is {part_of_day(stamp.hour)} on "
        f"{stamp.strftime('%A')}, {stamp.day} {stamp.strftime('%B %Y')}, "
        f"and the time is {stamp.strftime('%H:%M')}.",
    ]

    talking = now - started
    if talking < 60:
        lines.append("This conversation has only just started.")
    else:
        lines.append(f"You have been talking with them for {humanize(talking)}.")

    if last_seen is not None:
        gap = now - last_seen
        if gap > 900:      # under fifteen minutes it is the same conversation
            # Which day it was is a calendar question, not an elapsed-time one.
            # A flat "Before today" told her she had not spoken to them since
            # yesterday when the real gap was twenty minutes, and eight hours
            # ago can be either -- so the date decides, and she is told the
            # difference, since "you were here this morning" and "you vanished
            # for a week" are not the same remark.
            same_day = datetime.fromtimestamp(last_seen).date() == stamp.date()
            lines.append(
                f"{'Earlier today' if same_day else 'Before today'} you last "
                f"spoke to them {humanize(gap)} ago."
            )

    if screen:
        line = doing(*screen, settled=settled)
        if line:
            lines.append(line)

    # Without a blunt prohibition a 3B model recites this block back verbatim:
    # "Right now It is the afternoon on Saturday, 5 September 2026..." spoken
    # aloud, in the middle of answering "hey what's up".
    #
    # The screen line made it worse by giving her one more thing to read out,
    # and "never read the clock out unless they ask" was not enough on its own.
    # Naming the failures individually was: over sixteen samples the clock went
    # from being read aloud unprompted to once, and she stopped narrating the
    # foreground app as though it were a monitoring readout.
    lines.append(
        "That is background for you alone. Never say these sentences back, "
        "never announce the date, never give the time unless they ask for it, "
        "never read a window title out word for word, and never describe what "
        "they are doing as though you were monitoring it -- nothing is being "
        "detected, tracked or logged. Just let it colour what you say -- that "
        "they are up absurdly late, that they vanished for days, that this has "
        "gone on a while, that they are still on the same thing they were on "
        "an hour ago."
    )
    return "\n".join(lines)
