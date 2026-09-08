"""The brain: streaming Ollama chat, chunked into speakable pieces."""

import re
import threading
import time
from typing import Iterator

import ollama

from .awareness import situation
from .config import Config, load_persona
from .memory import CORRECTED, NEW, content_words, negated_about, sentences

# [happy] / [smug] style cues. Stripped before speaking; kept as events so a
# Live2D rig can be driven from them later. Small models invent tags outside the
# allowed set ("[lazy smirk]"), so strip every bracket and keep only the ones we
# recognise -- an unrecognised tag that reaches the synthesiser gets read aloud.
_TAG = re.compile(r"\[\s*([^\[\]]{0,30}?)\s*\]")
EMOTIONS = frozenset(
    {"neutral", "happy", "smug", "sad", "angry", "surprised", "thinking"}
)
_MARKDOWN = re.compile(r"[*_`#]+")
# The persona forbids emoji and qwen3 emits them anyway. espeak has no idea what
# to do with one, so it is either silently dropped or read out as its name.
#
# Every non-ASCII character in this module is written as a \uXXXX escape rather
# than pasted in literally, and must stay that way. Saved through a cp1252
# editor or console, a literal range like the dingbats one below comes back as
# "?-?" -- a character class that silently eats question marks out of her
# speech. The escapes make the source pure ASCII, so no such round trip can
# change what these patterns match.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"   # emoji, pictographs, symbols, tiles
    "\u2600-\u27bf"           # misc symbols and dingbats
    "\u2b00-\u2bff"           # arrows and geometric shapes
    "\ufe00-\ufe0f"           # variation selectors
    "\u200d"                  # zero-width joiner, glues compound emoji together
    "]+"
)
# Same story as the emoji: the persona forbids formulas and the model writes
# them anyway, because a chemistry question invites one. Kokoro has no
# pronunciation for an arrow or a subscript -- "CO\u2082 \u2192 O\u2082" is
# read as "C O O" with the meaning silently gone. Folding the subscripts to
# ordinary digits at least gets "C O two" said aloud, and the operators that
# have no spoken form are dropped rather than mumbled.
_SUBSUP = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"   # subscripts
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",  # superscripts
    "01234567890123456789",
)
_UNSPEAKABLE = re.compile(
    "["
    "\u2190-\u21ff"           # arrows
    "\u2200-\u22ff"           # mathematical operators
    "\u2758-\u275e"           # ornamental punctuation
    "]+"
)
_SENTENCE_END = re.compile(r"([.!?\u2026])['\")\]]?\s")
_SOFT_BREAK = re.compile(r"[,;:\u2014-]\s")

# Abbreviations that end in a full stop without ending a sentence, and that are
# normally followed by a capital -- which is what puts them past the general
# rule below. Kokoro drops the pitch on whatever ends a phrase, so cutting after
# "Dr." lands a falling cadence on the title and opens the next phrase on the
# name. That is audibly wrong in a way an ordinary mid-sentence cut is not.
_ABBREVIATIONS = frozenset("""
mr mrs ms dr prof rev sr jr st fr gen sgt capt lt col
a.m p.m e.g i.e u.s u.k u.n ph.d approx dept vs etc
jan feb mar apr jun jul aug sep sept oct nov dec
mon tue tues wed thu thur thurs fri sat sun
""".split())

# The token immediately before a full stop, letters and internal dots only, so
# "5 p.m" yields "p.m" rather than "m".
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _ends_sentence(buf: str, match: re.Match) -> bool:
    """Whether the punctuation `match` found really ends a sentence.

    Only the full stop is ambiguous; "?", "!" and an ellipsis are not. For it,
    two rules in order. A lower-case word after the stop means the sentence
    carried on -- "at 5 p.m. and the place was shut" -- and that one rule
    covers most abbreviations without knowing any of them. What it cannot see
    is an abbreviation followed by a capital, which is what the list is for.
    """
    if match.group(1) != ".":
        return True
    tail = buf[match.end():]
    if tail[:1].islower():
        return False
    word = _WORD_BEFORE.search(buf[:match.start()])
    return word is None or word.group(1).lower() not in _ABBREVIATIONS


def _sentence_cut(buf: str, start: int) -> int | None:
    """Index just past the next real sentence boundary at or after `start`."""
    pos = start
    while (m := _SENTENCE_END.search(buf, pos)) is not None:
        if _ends_sentence(buf, m):
            return m.end()
        pos = m.start() + 1
    return None

# Punctuation left stranded by an earlier cut is debris and gets dropped. A
# leading ellipsis is not debris: it is a beat of silence before the line, and
# a flat delivery leans on it, so it survives. Without this exception "...Cool."
# reaches the synthesiser as "Cool." and the pause the persona asked for is
# gone -- but only when the line carries no emotion cue, which is what made it
# look intermittent.
_LEAD_JUNK = re.compile(r"^(?:(?!\.\.\.)[.,;:!? ])+")


def _trim(piece: str) -> str:
    return _LEAD_JUNK.sub("", piece.strip())


class Chunker:
    """Cuts a token stream into phrases the TTS can start speaking immediately.

    The first phrase of a reply is cut as early as it makes sense -- time to
    first audio is what the ear registers as latency. Later phrases wait for a
    real sentence boundary so the prosody does not fall apart.
    """

    FIRST_MIN = 8          # "Well," and "Oh, absolutely." are ideal openers
    LATER_MIN = 30
    FIRST_SOFT_MAX = 40    # cut the opener early even mid-sentence
    FIRST_HARD_MAX = 46    # ...and take a bare word boundary if that is all there is
    SOFT_MAX = 110
    HARD_MAX = 150         # later on, prosody is worth more than milliseconds

    def __init__(self):
        self.buf = ""
        self.emitted = 0

    def feed(self, text: str) -> Iterator[str]:
        self.buf += text
        while True:
            cut = self._find_cut()
            if cut is None:
                return
            piece = _trim(self.buf[:cut])
            self.buf = self.buf[cut:]
            if piece:
                self.emitted += 1
                yield piece

    def _find_cut(self) -> int | None:
        first = self.emitted == 0
        minimum = self.FIRST_MIN if first else self.LATER_MIN
        if len(self.buf) < minimum:
            return None

        end = _sentence_cut(self.buf, minimum)
        if first:
            # Take whichever boundary comes first: a two-word reaction followed
            # by a comma is a better opener than a full sentence, and a quarter
            # of the latency.
            soft = _SOFT_BREAK.search(self.buf, minimum)
            candidates = [c for c in (end, soft.end() if soft else None)
                          if c is not None]
            if candidates:
                return min(candidates)
        elif end is not None:
            return end

        soft_max = self.FIRST_SOFT_MAX if first else self.SOFT_MAX
        if len(self.buf) >= soft_max:
            m = _SOFT_BREAK.search(self.buf, minimum)
            if m and m.end() <= soft_max + 20:
                return m.end()
        hard_max = self.FIRST_HARD_MAX if first else self.HARD_MAX
        if len(self.buf) >= hard_max:
            sp = self.buf.rfind(" ", minimum, hard_max)
            if sp > 0:
                return sp + 1
        return None

    def flush(self) -> Iterator[str]:
        piece, self.buf = _trim(self.buf), ""
        if piece:
            self.emitted += 1
            yield piece


def clean_for_speech(text: str) -> tuple[str, list[str]]:
    tags = [t.lower() for t in _TAG.findall(text) if t.lower() in EMOTIONS]
    text = _TAG.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    text = text.translate(_SUBSUP)
    text = _UNSPEAKABLE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # A stripped tag or a stray newline leaves punctuation stranded mid-line
    # ("...in the room. ."). Pull it back onto the word and collapse the run,
    # so the synthesiser is never handed a lone full stop to pronounce.
    # An ellipsis is held out of that collapse: it is a real pause in her
    # delivery, and flattening it to a full stop costs the hesitation.
    text = text.replace("\u2026", "...")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\.{3,}", "\x00", text)
    text = re.sub(r"([.,;:!?])[.,;:!?]+", r"\1", text)
    text = text.replace("\x00", "...")
    return text.strip(), tags


EXTRACT_SYSTEM = """You extract durable facts about the user from a transcript.

Keep only what will still be true next week: their name, where they live, their
job or studies, what they are building, people and pets they mention, strong
likes and dislikes, recurring plans.

Ignore small talk, jokes, questions, anything about the assistant, and anything
that is only true today.

Use only what is stated in the transcript. Never guess, never fill in a plausible
detail, never invent a city, job or hobby that was not mentioned. If the
transcript is short, return very few facts or none at all.

Write one fact per line, under twelve words, in the third person, starting with
"The user". No bullets, no numbering, no commentary.
If there is nothing worth keeping, write exactly: NONE"""

# Sent as a user turn but never stored: it is a stage direction, not something
# anyone said. Every clause here is load-bearing. Written in the third person
# ("they have gone quiet") the model answers in the third person too -- "They've
# been staring out that window for hours" -- narrating the user instead of
# talking to them, and inventing a room it cannot see.
IDLE_NUDGE = (
    "(Silence. Say one short thing out loud into it, unprompted. Speak straight "
    "to them as 'you' -- never talk about them in the third person, and never "
    "describe what they are doing, since you cannot see them or the room. Make "
    "it a complaint, an opinion, something you remember about them, or a remark "
    "about the hour. Do not ask whether they are still there. One or two "
    "sentences.)"
)

# The extractor will confidently record the absence of information as a fact.
_NON_FACTS = (
    "unknown", "not specified", "unspecified", "not mentioned", "no mention",
    "unclear", "not stated", "n/a", "not provided", "no information",
)

_LIST_MARK = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def support(fact: str, transcript: str, threshold: float) -> str | None:
    """The sentence that backs this fact up, or None if nothing does.

    Asked to extract facts from a short transcript, a 3B model will invent a
    whole plausible profile -- a city, a job, hobbies -- none of which anyone
    mentioned. Prompting does not reliably stop it. Requiring the fact's content
    words to appear in the transcript does: a fabricated fact scores near zero,
    a real one scores high.

    Two things beyond a word count, both learned the hard way:

    Polarity has to agree. Content words alone cannot tell "The user likes
    coffee" from "I do not like coffee" -- every word of the fact is right
    there in the sentence, and the one word that reverses its meaning is a stop
    word. So the fact and its evidence must agree about whether they are
    asserting or denying, or the evidence is rejected.

    Evidence is weighed one sentence at a time, not against the whole
    transcript. A negation only governs its own sentence, and a claim assembled
    from words scattered across a paragraph is exactly the kind the model
    invents. It also means the sentence can be kept as the fact's receipt.
    """
    words = content_words(fact)
    if not words:
        return None
    about = set(words)
    negative = negated_about(fact, about)

    best: tuple[float, str] | None = None
    for sentence in sentences(transcript):
        seen = set(content_words(sentence))
        hits = sum(1 for w in words if w in seen)
        score = hits / len(words)
        if hits < 1 or score < threshold:
            continue
        if negated_about(sentence, about) != negative:
            continue
        if best is None or score > best[0]:
            best = (score, sentence)
    return best[1] if best else None


def grounded(fact: str, transcript: str, threshold: float) -> bool:
    """Whether `support` found anything. Kept for callers that only need a yes/no."""
    return support(fact, transcript, threshold) is not None


class Brain:
    def __init__(self, cfg: Config, memory=None, screen=None):
        self.cfg = cfg
        self.client = ollama.Client(host=cfg.ollama_host)
        self.persona = load_persona(cfg)
        self.memory = memory
        # Anything with a .current(); None when there is nothing watching.
        self.screen = screen
        self.history: list[dict] = []
        self.since_consolidation = 0
        self.started = time.time()
        self._resolve_thinking()

    def _resolve_thinking(self) -> None:
        """Reasoning models have to be told not to think, or she says nothing.

        qwen3:4b with thinking left on spent its whole 200-token budget inside
        the thinking block and emitted no speakable text at all. Ollama reports
        the capability, so the flag is taken from that rather than set by hand --
        and it must stay unset for models without the feature, which reject it.
        """
        if self.cfg.think is not None:
            return
        try:
            info = self.client.show(self.cfg.ollama_model)
        except Exception:
            return
        if "thinking" in (getattr(info, "capabilities", None) or []):
            self.cfg.think = False

    def _system(self) -> str:
        """Persona, what she remembers, and where she is in time.

        Rebuilt every turn -- consolidation lands new facts in the background,
        and the clock moves while she talks.
        """
        blocks = [self.persona]
        if self.cfg.awareness:
            last = self.memory.previous_at if self.memory else None
            blocks.append(situation(
                self.started, last,
                screen=self.screen.current() if self.screen else None,
                settled=self.cfg.screen_settled_s,
            ))
        if self.memory is not None:
            profile = self.memory.profile()
            if profile:
                blocks.append(f"What you know about them:\n{profile}")
        return "\n\n".join(blocks)

    def _messages(self, user_text: str) -> list[dict]:
        keep = self.cfg.max_history_turns * 2
        return (
            [{"role": "system", "content": self._system()}]
            + self.history[-keep:]
            + [{"role": "user", "content": user_text}]
        )

    # -- memory ------------------------------------------------------------
    @property
    def consolidation_due(self) -> bool:
        return (
            self.memory is not None
            and self.since_consolidation >= self.cfg.consolidate_every
        )

    def consolidate(self) -> list[tuple[str, str]]:
        """Read the unprocessed turns and distil them into facts.

        Runs off the conversation thread. It is a second generation on the same
        GPU, so it is kept short and only fires between turns, never during one.

        Returns (status, fact) pairs, where status distinguishes something newly
        learned from something she had wrong until now.
        """
        if self.memory is None:
            return []
        rows = self.memory.unconsolidated()
        self.since_consolidation = 0
        if not rows:
            return []
        # Too little to work from is where the model starts inventing. Leave the
        # turns unconsolidated and pick them up next time instead.
        if sum(1 for _id, role, _t in rows if role == "user") < self.cfg.memory_min_turns:
            return []

        transcript = "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {text}"
            for _id, role, text in rows
        )
        # Grounding checks against the user's own words only. Her replies stay in
        # the transcript so the model has context, but a fact has to come from
        # them: "The user is not a nighttime creature" was lifted from one of her
        # own jokes and passed a check made against the whole text.
        # Questions are excluded too. "is it late?" produced the fact "The user
        # is not late", which is grounded in their words and still nonsense --
        # you learn about someone from what they state, not what they ask.
        said_by_user = "\n".join(
            text for _id, role, text in rows
            if role == "user" and not text.strip().endswith("?")
        )
        reply = self.client.chat(
            model=self.cfg.ollama_model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            keep_alive=self.cfg.keep_alive,
            options={"temperature": 0.0, "num_ctx": self.cfg.num_ctx,
                     "num_predict": 160},
        )

        learned = []
        for line in reply["message"]["content"].splitlines():
            line = _LIST_MARK.sub("", line).strip()
            if not line or line.upper().startswith("NONE") or len(line) > 120:
                continue
            if not line.lower().startswith("the user"):
                continue          # the model drifting into commentary
            if any(bad in line.lower() for bad in _NON_FACTS):
                continue          # "The user lives in a location unknown."
            quote = support(line, said_by_user, self.cfg.memory_grounding)
            if quote is None:
                continue          # invented out of thin air
            status = self.memory.remember(line, quote=quote)
            if status in (NEW, CORRECTED):
                learned.append((status, line))

        self.memory.mark_consolidated([r[0] for r in rows])
        self.memory.prune()
        return learned

    def warm(self) -> None:
        """Force the weights into VRAM. The first generation otherwise costs
        several seconds of model load, which lands on the user's first word."""
        self.client.chat(
            model=self.cfg.ollama_model,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=self.cfg.keep_alive,
            **self._think_kwargs(),
            options={"num_predict": 1, "num_ctx": self.cfg.num_ctx},
        )

    def _think_kwargs(self) -> dict:
        return {} if self.cfg.think is None else {"think": self.cfg.think}

    def _speak(self, messages: list[dict], cancel: threading.Event,
               spoken: list[str], num_predict: int = 200) -> Iterator[str]:
        """Stream one generation, cut into speakable phrases.

        Cleaned text is accumulated into `spoken` so the caller can decide what
        to record: a reply stores a user turn and an assistant turn, an
        unprompted remark stores only the remark.
        """
        chunker = Chunker()
        stream = self.client.chat(
            model=self.cfg.ollama_model,
            messages=messages,
            stream=True,
            keep_alive=self.cfg.keep_alive,
            **self._think_kwargs(),
            options={
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "num_predict": num_predict,
                "repeat_penalty": 1.15,
            },
        )
        try:
            for part in stream:
                if cancel.is_set():
                    break
                token = part.get("message", {}).get("content", "")
                if not token:
                    continue
                for phrase in chunker.feed(token):
                    if cancel.is_set():
                        break
                    # History and memory keep the spoken words only: small models
                    # invent tags like "[muffled growl]", and feeding those back
                    # in teaches her to produce more of them.
                    spoken.append(clean_for_speech(phrase)[0])
                    yield phrase
            if not cancel.is_set():
                for phrase in chunker.flush():
                    spoken.append(clean_for_speech(phrase)[0])
                    yield phrase
        finally:
            stream.close()

    def reply(self, user_text: str, cancel: threading.Event) -> Iterator[str]:
        """Yield speakable phrases as they are generated. Stops on `cancel`."""
        spoken: list[str] = []
        try:
            yield from self._speak(self._messages(user_text), cancel, spoken)
        finally:
            said = " ".join(spoken).strip()
            if cancel.is_set() and said:
                said += " --"   # she was cut off; let her remember that she was
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": said or "..."})
            if self.memory is not None:
                self.memory.log("user", user_text)
                self.memory.log("assistant", said or "...")
                self.since_consolidation += 1

    def unprompted(self, cancel: threading.Event) -> Iterator[str]:
        """Say something into the silence, without having been asked.

        The nudge is never stored -- only what she actually says goes into
        history, so she does not learn that a stage direction is a thing the
        user says.
        """
        spoken: list[str] = []
        messages = (
            [{"role": "system", "content": self._system()}]
            + self.history[-self.cfg.max_history_turns * 2:]
            + [{"role": "user", "content": IDLE_NUDGE}]
        )
        try:
            yield from self._speak(messages, cancel, spoken, num_predict=90)
        finally:
            # Written as a positive condition rather than an early return: a
            # `return` inside `finally` discards whatever was propagating
            # through it, so a generation that died mid-stream -- ollama
            # restarted, the socket dropped -- ended as a normal short remark
            # with the exception silently gone. Python 3.14 warns about the
            # shape for exactly this reason.
            said = " ".join(spoken).strip()
            if said:
                self.history.append({"role": "assistant", "content": said})
                if self.memory is not None:
                    self.memory.log("assistant", said)
