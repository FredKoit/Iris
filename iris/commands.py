"""Spoken commands that act on memory directly, instead of being replied to.

"Remember that I hate mushrooms" is not conversation, it is an instruction, and
routing it through the extractor is the wrong shape twice over: the extractor
only runs every few turns, and it is free to decide the sentence was not worth
keeping. Something asked for out loud should land immediately and exactly.

Everything here is a pure function of a string, so it can be tested without a
microphone.
"""

import re

REMEMBER = "remember"
FORGET = "forget"

# Shared with timers.py: "okay iris can you ... please" wraps any spoken
# command, not just a memory one.
# Nobody says "forget the microwave." They say "okay, can you forget about the
# microwave please?" -- which is an instruction wearing a question mark. This
# preamble is what separates the two, and it is why a trailing "?" alone cannot
# be trusted to mean someone is asking rather than telling.
POLITE = (
    r"(?:(?:ok|okay|alright|right|yeah)[,\s]+)?"
    r"(?:hey\s+)?(?:iris[,\s]+)?"
    r"(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?"
)
TRAILING = r"\s*(?:[,\s]*please)?\s*[.!?]*\s*$"

# What separates the verb from what it acts on. A comma counts: Whisper puts one
# after a leading imperative constantly, and requiring bare whitespace left
# "remember, I hate mushrooms" -- an entirely ordinary thing to say -- silently
# unrecognised, to be answered conversationally instead of stored.
_SEP = r"(?:\s*[,:]\s*|\s+)"

# "remember X" / "note that X" / "keep in mind X". Anchored at the start, so an
# ordinary sentence that merely contains the word ("I remember going there") is
# left alone. The \b matters as much as the anchor: without it "remember"
# matched inside "remembering", and the participle is never an instruction.
_REMEMBER = re.compile(
    r"^\s*" + POLITE +
    r"(?:(?:remember|keep\s+in\s+mind)\b" + _SEP +
    # "note" is a noun at least as often as it is a verb, and unlike the others
    # it reads as an instruction only when something joins it to what follows.
    # Bare, it swallowed "note taking is hard" and filed it as a fact.
    r"|note\b(?:\s*[,:]\s*|\s+that\s+))"
    r"(?:that\s+|this[:,]\s*|about\s+)?"
    r"(?P<what>.+?)" + TRAILING,
    re.I,
)

# "forget that" / "forget about X" / "stop talking about X". The separator is
# optional here and only here, because a bare "forget" is a whole command --
# it means the thing she just learned.
_FORGET = re.compile(
    r"^\s*" + POLITE +
    r"(?:forget|stop\s+(?:talking|going\s+on)\s+about)\b" + _SEP + r"?"
    r"(?:about\b\s*|that\b\s*|the\s+fact\s+that\b\s*"
    r"|what\s+i\s+said\s+about\b\s*)?"
    r"(?P<what>.*?)" + TRAILING,
    re.I,
)

# "remember when we..." is reminiscing and "how I take my coffee" is a clause,
# not a fact. Anything opening with one of these is left for the brain.
_NOT_A_COMMAND = re.compile(
    r"^(?:when|if|how|why|what|where|who|whether|the\s+time|that\s+time)\b", re.I
)

# "forget it" is a shrug. Wiping memory because someone dropped a subject would
# be an unpleasant surprise, so only an explicit target -- or a bare "forget
# that", which means the thing she just learned -- counts.
_SHRUG = frozenset(["it", "never mind", "nevermind", "everything i said"])

# Rewrites of the speaker's own words into the third person the fact store uses,
# so an explicit fact reads like an extracted one and can be compared with it.
# Order matters: the contractions have to go before the bare "i".
_PRONOUNS = [
    (r"\bi\s+am\b", "the user is"),
    (r"\bi'm\b", "the user is"),
    (r"\bi\s+have\b", "the user has"),
    (r"\bi've\b", "the user has"),
    (r"\bi'll\b", "the user will"),
    (r"\bi'd\b", "the user would"),
    (r"\bmyself\b", "themselves"),
    (r"\bmy\b", "their"),
    (r"\bmine\b", "theirs"),
    (r"\bme\b", "them"),
    (r"\bi\b", "the user"),
]

# Verbs that are already third person, or never take the -s.
_KEEP = frozenset("""
is was were has had can could will would shall should may might must does did
""".split())
_IRREGULAR = {
    "have": "has", "do": "does", "go": "goes", "am": "is", "are": "is",
    "be": "is", "don't": "doesn't", "dont": "doesnt",
    "haven't": "hasn't", "havent": "hasnt",
}
# Adverbs that sit between the subject and the verb, so agreement skips over
# them instead of inflecting the adverb ("the user reallys like coffee").
_ADVERBS = frozenset("""
always never usually often sometimes really still also only just rarely mostly
absolutely definitely genuinely secretly generally normally probably certainly
""".split())


def parse(text: str) -> tuple[str, str] | None:
    """Recognise a memory command. Returns (kind, payload), or None.

    For FORGET, an empty payload means "the most recent thing you learned".

    A trailing "?" is taken as a genuine question for REMEMBER but not for
    FORGET, and the asymmetry is deliberate. "Can you remember my name?" is
    usually asking her to recall it, so acting on it stores junk. "Can you
    forget about the microwave?" is never a question about her capabilities --
    and being unable to hear it is what let a subject she invented run for
    twenty turns after being asked twice to drop it.
    """
    stripped = text.strip()
    if not stripped:
        return None

    m = _FORGET.match(stripped)
    if m:
        what = m.group("what").strip()
        if what.lower() in _SHRUG or _NOT_A_COMMAND.match(what):
            return None
        return FORGET, what

    if stripped.endswith("?"):
        return None

    m = _REMEMBER.match(stripped)
    if m:
        what = m.group("what").strip()
        if not what or _NOT_A_COMMAND.match(what):
            return None
        return REMEMBER, what
    return None


def _inflect(verb: str) -> str:
    low = verb.lower()
    if low in _IRREGULAR:
        return _IRREGULAR[low]
    if low in _KEEP or low.endswith("s") or not verb.isalpha():
        return verb                       # already agrees, or not a plain verb
    if low.endswith(("sh", "ch", "x", "z", "o")):
        return verb + "es"
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _agree(text: str) -> str:
    """Fix subject-verb agreement after "I" became "the user".

    Without it "remember I hate mushrooms" stores "The user hate mushrooms",
    which reads as broken next to every extracted fact.
    """
    m = re.match(r"^(the user\s+)(.*)$", text, re.I | re.S)
    if not m:
        return text
    head, rest = m.groups()
    words = rest.split(" ")
    i = 0
    while i < len(words) and (
        words[i].lower() in _ADVERBS
        or (len(words[i]) > 3 and words[i].lower().endswith("ly"))
    ):
        i += 1
    if i >= len(words):
        return text
    words[i] = _inflect(words[i])
    return head + " ".join(words)


def third_person(text: str) -> str:
    """Turn "I hate mushrooms" into "The user hates mushrooms"."""
    out = text.strip().rstrip(".!")
    if not out:
        return out
    for pattern, replacement in _PRONOUNS:
        out = re.sub(pattern, replacement, out, flags=re.I)
    # "my birthday is in May" has become "their birthday ...", which is a fact
    # about them even though it does not open with the subject.
    out = re.sub(r"^their\b", "the user's", out, flags=re.I)
    if out.lower().startswith("the user "):
        out = _agree(out)
    elif not out.lower().startswith("the user"):
        # A fragment, or a fact about someone else. Attributing it to them
        # beats storing a bare dangling phrase.
        out = f"The user: {out}"
    return out[0].upper() + out[1:]
