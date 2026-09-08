"""Durable memory: what she still knows about you tomorrow.

Two tables. `turns` is the raw conversation log, which exists so consolidation
has something to read and so you can go back and see what she heard. `facts` is
what actually reaches the prompt -- a small, deduplicated set of statements
about you, reinforced when they come up again, and replaced when you contradict
one.

Every fact carries the words it was drawn from. The extractor is wrong often
enough that being able to see *why* she believes something is worth the few
bytes, and a fact you cannot trace is one you cannot argue with.

There is deliberately no vector store. The whole fact set is a few hundred
tokens, so it is cheaper to load all of it than to spend latency retrieving
part of it. Revisit that when `facts` outgrows the character budget by a lot.
"""

import re
import sqlite3
import threading
import time
import uuid

from .config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id      INTEGER PRIMARY KEY,
    norm    TEXT NOT NULL UNIQUE,
    text    TEXT NOT NULL,
    quote   TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    hits    INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS turns (
    id      INTEGER PRIMARY KEY,
    session TEXT NOT NULL,
    role    TEXT NOT NULL,
    text    TEXT NOT NULL,
    at      REAL NOT NULL,
    used    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS turns_unused ON turns (used, id);
"""

# What remember() did, so the caller can say so out loud.
NEW = "new"
KNOWN = "known"
CORRECTED = "corrected"

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")
_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")
# Closed up rather than replaced with a space, unlike every other punctuation
# mark: "don't" has to survive as one token "dont". Split into "don" + "t" it
# stops looking like a negation at all, and the polarity check below silently
# passes everything a contraction denies.
# Written as escapes, not literals: see the note in llm.py about what a cp1252
# round trip does to a character class.
_APOSTROPHE = re.compile(r"['\u2019\u02bc]")


def _norm(text: str) -> str:
    """Key used for deduplication. Punctuation and case carry no meaning here."""
    text = _APOSTROPHE.sub("", text.lower())
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def sentences(text: str) -> list[str]:
    """Split a transcript into the individual claims it makes.

    Grounding is checked one sentence at a time rather than against the whole
    bag of words, because a negation only governs its own sentence -- and
    because a fact whose evidence cannot be pinned to a single sentence has no
    quote worth storing.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


# Words that carry no evidence, so they are ignored when a fact is weighed
# against the words it supposedly came from, or against another fact. The
# negations in here are deliberate: polarity is a separate axis, compared
# explicitly below, and counting "not" as a content word would let the mere
# presence of a negation anywhere in the transcript pass as evidence.
_STOP = frozenset("""
the user a an and or of to in on at for with is are was were has have had
their they them his her its it that this these those be been being do does
did will would can could not no as by from about into over under new
""".split())

# Polarity markers. A fact and the sentence it came from have to agree on these.
# Without that check the content words of "I do not like coffee" fully support
# "The user likes coffee" -- every word of the fact appears in the transcript,
# and the only thing distinguishing them is the word the stop list threw away.
_NEGATORS = frozenset("""
not no never none nothing nobody dont doesnt didnt cant cannot wont
wouldnt shouldnt couldnt isnt arent wasnt werent aint neither nor without
hardly barely rarely stopped quit anymore
""".split())

# A negator governs the few words after it, not the whole sentence: in "I don't
# like coffee but I love tea" it applies to the coffee and not to the tea.
_CLAUSE_BREAK = frozenset("but however though although yet while whereas".split())
_NEGATION_WINDOW = 4


def _stem(word: str) -> str:
    """Strip a trailing plural / third-person "s", and nothing else.

    Not a stemmer and not trying to be. It exists so "likes" and "like" land on
    the same token, which is the one morphological difference that actually
    decides whether a correction is recognised as being about the same thing.
    Anything more aggressive starts folding words that mean different things.
    """
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def content_words(text: str) -> list[str]:
    """The words of `text` that carry its meaning, stemmed."""
    return [_stem(w) for w in _norm(text).split()
            if w not in _STOP and w not in _NEGATORS and len(w) > 2]


def topic(text: str) -> frozenset[str]:
    """What a fact is *about*, ignoring how it is phrased and whether it denies."""
    return frozenset(content_words(text))


def negation_scope(tokens: list[str]) -> set[int]:
    """Indices of the tokens that fall under a negation."""
    scope: set[int] = set()
    for i, token in enumerate(tokens):
        if token not in _NEGATORS:
            continue
        for j in range(i + 1, min(len(tokens), i + 1 + _NEGATION_WINDOW)):
            if tokens[j] in _CLAUSE_BREAK or tokens[j] in _NEGATORS:
                break
            scope.add(j)
    return scope


def negated_about(text: str, about: set[str] | frozenset[str]) -> bool:
    """Does `text` deny the words in `about`, rather than assert them?"""
    if not about:
        return False
    tokens = _norm(text).split()
    scope = negation_scope(tokens)
    if not scope:
        return False
    return any(i in scope and _stem(t) in about for i, t in enumerate(tokens))


def weight(hits: int, updated: float, now: float, half_life_days: float) -> float:
    """How much a fact counts, given how often and how recently it came up.

    Reinforcement alone was the whole ranking, and it had two failures. The
    prompt filled with whatever happened to get echoed rather than whatever
    mattered -- only about sixteen of forty facts fit the character budget, so
    the order decides what she actually knows. And once forty facts had each
    been reinforced even once, a newly learned fact arrived with a single hit,
    sorted below all of them, and was pruned before it could ever come up
    again: nothing new could stick.

    Decay runs from `updated`, which is set every time a fact is confirmed, so
    anything that keeps coming up never decays at all. The cost is deliberate
    and worth stating plainly: something genuinely dormant for a season does
    eventually fall out of a store this small.
    """
    if half_life_days <= 0:
        return float(hits)
    age_days = max(0.0, (now - updated) / 86400.0)
    return hits * (0.5 ** (age_days / half_life_days))


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap of two topics."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.memory_db.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across the reply thread and the consolidation
        # thread, serialised by a lock -- the write volume here is tiny.
        self.db = sqlite3.connect(str(cfg.memory_db), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()
        self.lock = threading.Lock()
        self.session = uuid.uuid4().hex[:12]
        self.last_fact_id: int | None = None
        # Captured before this session writes anything, so it really is the end
        # of the previous conversation rather than the start of this one.
        row = self.db.execute("SELECT MAX(at) FROM turns").fetchone()
        self.previous_at = row[0] if row else None

    def _migrate(self) -> None:
        """Add columns a database written by an older version is missing."""
        have = {r[1] for r in self.db.execute("PRAGMA table_info(facts)")}
        if "quote" not in have:
            self.db.execute("ALTER TABLE facts ADD COLUMN quote TEXT")

    # -- writing -----------------------------------------------------------
    def log(self, role: str, text: str) -> None:
        with self.lock:
            self.db.execute(
                "INSERT INTO turns (session, role, text, at) VALUES (?, ?, ?, ?)",
                (self.session, role, text, time.time()),
            )
            self.db.commit()

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """Word overlap between two normalised facts (Jaccard).

        The extractor happily produces "building a local voice AI called Iris"
        and "has a recurring plan to build a voice AI" from the same sentence.
        Exact-match dedup does not catch that; this does, without a model.
        """
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def remember(self, text: str, quote: str | None = None) -> str:
        """Store a fact. Returns NEW, KNOWN or CORRECTED.

        Three outcomes, and the third is the one that matters. Two statements
        about the same subject that disagree about whether it is true are not
        duplicates and not separate facts -- they are a fact and its
        correction, and the newer one replaces the older in place. Reinforcing
        instead (the old behaviour) taught her the opposite of what you had
        just told her, and made the stale fact harder to prune for having been
        "seen twice".
        """
        text = text.strip()
        norm = _norm(text)
        if not norm:
            return KNOWN
        about = topic(text)
        negative = negated_about(text, about)
        now = time.time()

        with self.lock:
            # Every match is collected before anything is written. Returning on
            # the first one left any *other* fact this statement contradicts
            # sitting in the store, so she could hold both sides of a
            # correction at once -- and which side survived depended on the
            # order SQLite happened to hand the rows over, which is why the
            # scan is ordered now.
            rows = self.db.execute(
                "SELECT id, norm, text FROM facts ORDER BY id"
            ).fetchall()
            same: int | None = None       # this fact, already known
            stale: list[int] = []         # facts this one contradicts
            for fid, other_norm, other_text in rows:
                other_about = topic(other_text)
                phrased_alike = (
                    other_norm == norm
                    or self._similar(other_norm, norm) >= self.cfg.memory_dedup
                )
                same_subject = (
                    _overlap(about, other_about) >= self.cfg.memory_contradict
                )
                if not (phrased_alike or same_subject):
                    continue
                if negated_about(other_text, other_about) != negative:
                    stale.append(fid)
                elif phrased_alike and same is None:
                    same = fid
                # Same subject, same polarity, different wording: two cats with
                # different names, not one fact stated twice. Left alone.

            if same is not None:
                # Already known. Any contradicting fact still goes: hearing it
                # again confirms which side is true, so the other side is not
                # something she should still be carrying.
                self.db.execute(
                    "UPDATE facts SET hits = hits + 1, updated = ? WHERE id = ?",
                    (now, same),
                )
                self._delete(stale)
                self.db.commit()
                self.last_fact_id = same
                return CORRECTED if stale else KNOWN

            if stale:
                # The oldest contradicted fact is rewritten in place so the
                # correction inherits its id and its position; the rest of what
                # it contradicts is now simply wrong, and goes.
                keep = stale[0]
                self.db.execute(
                    "UPDATE facts SET norm = ?, text = ?, quote = ?, "
                    "updated = ?, hits = 1 WHERE id = ?",
                    (norm, text, quote, now, keep),
                )
                self._delete(stale[1:])
                self.db.commit()
                self.last_fact_id = keep
                return CORRECTED

            cur = self.db.execute(
                "INSERT INTO facts (norm, text, quote, created, updated) "
                "VALUES (?, ?, ?, ?, ?)",
                (norm, text, quote, now, now),
            )
            self.db.commit()
            self.last_fact_id = cur.lastrowid
        return NEW

    def _delete(self, ids: list[int]) -> None:
        """Drop facts by id. Caller holds the lock and commits."""
        for fid in ids:
            self.db.execute("DELETE FROM facts WHERE id = ?", (fid,))

    def forget(self, fact_id: int) -> None:
        with self.lock:
            self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self.db.commit()

    def forget_like(self, text: str, threshold: float | None = None) -> list[str]:
        """Drop every fact about the same subject as `text`. Returns what went.

        Deliberately matched on subject rather than wording: "forget about the
        coffee" has to reach "The user drinks too much coffee" without being
        phrased anything like it.
        """
        about = topic(text)
        if not about:
            return []
        limit = self.cfg.memory_forget if threshold is None else threshold
        with self.lock:
            doomed = [
                (fid, other)
                for fid, other in self.db.execute("SELECT id, text FROM facts")
                if _overlap(about, topic(other)) >= limit
            ]
            for fid, _text in doomed:
                self.db.execute("DELETE FROM facts WHERE id = ?", (fid,))
            self.db.commit()
        return [text for _fid, text in doomed]

    def forget_last(self) -> str | None:
        """Drop the most recently learned or reinforced fact."""
        with self.lock:
            row = self.db.execute(
                "SELECT id, text FROM facts ORDER BY updated DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self.db.execute("DELETE FROM facts WHERE id = ?", (row[0],))
            self.db.commit()
        return row[1]

    def _ranked(self) -> list[tuple]:
        """Every fact, strongest first. Caller holds the lock.

        Sorted in Python rather than SQL because the score is an exponential in
        wall-clock age, and the store is forty rows -- there is nothing here to
        optimise, and a decay curve written in SQLite CASE arithmetic would be
        unreadable for no gain.
        """
        rows = self.db.execute(
            "SELECT id, text, hits, quote, updated FROM facts"
        ).fetchall()
        now = time.time()
        half = self.cfg.memory_half_life_days
        rows.sort(key=lambda r: (weight(r[2], r[4], now, half), r[4]), reverse=True)
        return rows

    def prune(self) -> int:
        """Drop the weakest facts once the store outgrows its cap."""
        with self.lock:
            ranked = self._ranked()
            doomed = [r[0] for r in ranked[self.cfg.memory_max_facts:]]
            self._delete(doomed)
            self.db.commit()
            return len(doomed)

    # -- reading -----------------------------------------------------------
    def facts(self, limit: int | None = None) -> list[tuple[int, str, int]]:
        with self.lock:
            ranked = self._ranked()
        cap = limit or self.cfg.memory_max_facts
        return [(fid, text, hits) for fid, text, hits, _q, _u in ranked[:cap]]

    def detail(self, limit: int | None = None
               ) -> list[tuple[int, str, int, str, float]]:
        """Facts with their receipts and their score, for `scripts/memory.py`."""
        with self.lock:
            ranked = self._ranked()
        now = time.time()
        half = self.cfg.memory_half_life_days
        cap = limit or self.cfg.memory_max_facts
        return [
            (fid, text, hits, quote or "", weight(hits, updated, now, half))
            for fid, text, hits, quote, updated in ranked[:cap]
        ]

    def profile(self) -> str:
        """The memory block injected into the system prompt, within budget."""
        out, used = [], 0
        for _id, text, _hits in self.facts():
            if used + len(text) + 3 > self.cfg.memory_char_budget:
                break
            out.append(f"- {text}")
            used += len(text) + 3
        return "\n".join(out)

    def unconsolidated(self, limit: int = 40) -> list[tuple[int, str, str]]:
        with self.lock:
            return list(
                self.db.execute(
                    "SELECT id, role, text FROM turns WHERE used = 0 "
                    "ORDER BY id LIMIT ?",
                    (limit,),
                )
            )

    def mark_consolidated(self, ids: list[int]) -> None:
        if not ids:
            return
        with self.lock:
            self.db.executemany(
                "UPDATE turns SET used = 1 WHERE id = ?", [(i,) for i in ids]
            )
            self.db.commit()

    def keywords(self, limit: int = 24) -> list[str]:
        """Proper nouns worth biasing the transcriber toward.

        Names are exactly what a generic English model mishears, and the fact
        table already holds the ones that matter: the user's name, their pets,
        their projects. The longer she knows someone, the better she hears them.
        """
        words: list[str] = []
        seen = set()
        for _id, text, _hits in self.facts():
            body = text[8:] if text.lower().startswith("the user") else text
            for raw in re.findall(r"[A-Za-z][A-Za-z0-9+#.]*", body):
                low = raw.lower()
                if len(raw) < 3 or low in seen:
                    continue
                # Any internal capital marks a name or a piece of jargon
                # ("Pixel", "JavaScript"); the template text is all lowercase.
                if raw != low:
                    seen.add(low)
                    words.append(raw.rstrip("."))
        return words[:limit]

    def stats(self) -> dict:
        with self.lock:
            facts = self.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            turns = self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            pending = self.db.execute(
                "SELECT COUNT(*) FROM turns WHERE used = 0"
            ).fetchone()[0]
        return {"facts": facts, "turns": turns, "pending": pending}

    def close(self) -> None:
        with self.lock:
            self.db.close()
