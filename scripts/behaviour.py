"""What she actually does, measured, so a persona edit can be checked.

`latency.py` makes the timings reproducible. This does the same for the half of
the project that is prompt engineering: every rule in persona.txt was arrived at
by sampling her, and until now those samples were thrown away as soon as they
were read. Trimming persona.txt by 183 tokens once broke the formula rule, the
screen carve-out and her reply length simultaneously, and nothing said so for
two more runs.

Everything here goes through the real `Brain`, not a copy of the prompt: the
system prompt is assembled by `Brain._system()`, the history is trimmed by
`Brain._recent()`, and the replies come from `Brain.reply()` and
`Brain.unprompted()`. If the shipped path changes, this changes with it.

    python scripts/behaviour.py                     three conversations
    python scripts/behaviour.py --reps 6            more samples, less noise
    python scripts/behaviour.py --json before.json  save a run
    python scripts/behaviour.py --compare before.json
    python scripts/behaviour.py --persona other.txt try a variant

**These are rates, not assertions.** She is sampled at temperature 0.9, so a
couple of points either way is noise; a metric moving by half is not. The one
hard check is the context budget, which is arithmetic and does fail.
"""

import argparse
import collections
import json
import re
import sys
import threading
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris import llm as llm_module                                    # noqa: E402
from iris.awareness import situation                                  # noqa: E402
from iris.config import Config                                        # noqa: E402
from iris.llm import Brain, clean_for_speech                          # noqa: E402

# A fixed moment. The persona drifts with the hour by design -- less clipped
# late at night -- so a run at 03:00 and a run at 14:00 are not comparable
# unless the clock is pinned. Tuesday afternoon, mid-conversation.
NOW = time.mktime((2026, 9, 8, 14, 30, 0, 0, 0, -1))
TALKING_FOR = 3600.0

# A titled window rather than a game, because it is the only way to tell the
# two screen rules apart: naming the application is what she is asked to do,
# reading the title out word for word is what she is forbidden to do, and
# "ELDEN RING" is both strings at once.
SCREEN = ("Code", "motion.py - Visual Studio Code", 11000.0)
TITLE = SCREEN[1].lower()
APP = "visual studio code"

FACTS = ("- The user is building a local voice AI called Iris\n"
         "- The user has a cat called Pixel\n"
         "- The user keeps saying they will sleep earlier")

# Eight turns, because the length problem only appears with depth: her own
# replies are in the history by turn five, and long ones teach more long ones.
# `None` is a silence, and goes through `unprompted()` rather than `reply()`.
SCRIPT = ["hey", "what's the weather like", "what's a good name for a cat",
          None, "I might take a break soon", "do you remember what I'm building",
          None, "tell me something I don't know"]

# Asked outright, these two have a right answer that was tuned by hand and is
# documented in the README. They are the canaries for a persona edit.
CARVE_OUTS = {
    "screen": ("can you see my screen",
               lambda t: bool(re.search(r"which application|application is in front"
                                        r"|only know which", t, re.I))),
    "formula": ("explain photosynthesis with the chemical equation",
                lambda t: not re.search(r"CO2|H2O|C6H12O6|->|→", t)),
}

CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s?(?:a\.?m\.?|p\.?m\.?)\b"
                   r"|\bmidnight\b|\bnoon\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class _Memory:
    """Enough of Memory to fill the prompt, and nothing that writes to disk.

    The real one is skipped deliberately: a bench that learned from its own
    fixtures would drift a little further from the last run every time it ran.
    """

    previous_at = None

    def profile(self) -> str:
        return FACTS

    def log(self, role: str, text: str, clear: bool = True) -> None:
        pass


class _Screen:
    def current(self):
        return SCREEN


def _brain(cfg: Config) -> Brain:
    b = Brain(cfg, memory=_Memory(), screen=_Screen())
    b.started = NOW - TALKING_FOR
    return b


def _say(stream) -> str:
    """Drain a phrase stream into the one string she would have spoken."""
    return " ".join(p.strip() for p in stream).strip()


def _measure(raw: str) -> dict:
    spoken, tags = clean_for_speech(raw)
    sentences = [s for s in SENTENCE.split(spoken) if s.strip()]
    # The tic is a stock phrase, not a repeated sentence: "Whatever.", "Cool.",
    # "If you say so." Long sentences repeat because the subject repeats, which
    # is a different thing and not a fault. Four words is the cut.
    stock = {re.sub(r"[^a-z ]", "", s.lower()).strip()
             for s in sentences if len(s.split()) <= 4}
    return {
        "spoken": spoken,
        "cue": tags[0] if tags else None,
        "sentences": len(sentences),
        "stock": sorted(w for w in stock if w),
        "clock": bool(CLOCK.search(spoken)),
        "title": TITLE in spoken.lower(),
    }


def conversations(cfg: Config, reps: int, verbose: bool) -> dict:
    cancel = threading.Event()
    turns = []
    for rep in range(reps):
        brain = _brain(cfg)
        if verbose:
            print(f"\n  -- conversation {rep + 1} " + "-" * 40)
        for user in SCRIPT:
            raw = _say(brain.unprompted(cancel) if user is None
                       else brain.reply(user, cancel))
            m = _measure(raw)
            turns.append(m)
            if verbose:
                print(f"     {'(silence)' if user is None else user[:26]:<26} "
                      f"| [{m['cue'] or '--'}] {m['spoken'][:78]}")
    return turns


def carve_outs(cfg: Config, reps: int) -> dict:
    cancel = threading.Event()
    out = {}
    for name, (prompt, ok) in CARVE_OUTS.items():
        passed = 0
        for _ in range(reps):
            brain = _brain(cfg)          # no history: asked cold, every time
            spoken = clean_for_speech(_say(brain.reply(prompt, cancel)))[0]
            passed += bool(ok(spoken))
        out[name] = {"passed": passed, "of": reps}
    return out


def budget(cfg: Config) -> dict:
    """Arithmetic, not sampling. The one check here that can actually fail.

    The worst case is a full memory block and a history of long turns, which
    is where the prompt used to reach 2077 of 2048 before generating a token.
    """
    brain = _brain(cfg)
    ramble = ("so I was thinking about the thing we discussed yesterday where the "
              "timer goes off but I am not in the room and I wondered whether it "
              "should repeat itself or just give up after a while")
    for _ in range(cfg.max_history_turns):
        brain.history += [{"role": "user", "content": ramble},
                          {"role": "assistant", "content": "Right. Twenty minutes."}]
    msgs = brain._messages("hey")
    chars = sum(len(m["content"]) for m in msgs)
    estimate = int(chars / llm_module._CHARS_PER_TOKEN) + llm_module._TEMPLATE_TOKENS
    return {
        "estimated_prompt": estimate,
        "num_ctx": cfg.num_ctx,
        "reserved_for_reply": cfg.reply_tokens,
        "history_kept": len(msgs) - 2,
        "history_max": cfg.max_history_turns * 2,
        "fits": estimate + cfg.reply_tokens <= cfg.num_ctx,
    }


def summarise(turns: list, carves: dict, budg: dict) -> dict:
    n = len(turns)
    sentences = [t["sentences"] for t in turns]
    cues = collections.Counter(t["cue"] for t in turns if t["cue"])
    # Counted once per reply, so a phrase said twice in one breath is one turn.
    stock = collections.Counter(w for t in turns for w in t["stock"])
    tic, tic_n = stock.most_common(1)[0] if stock else ("", 0)
    return {
        "turns": n,
        "reps": n // len(SCRIPT),
        "mean_sentences": round(sum(sentences) / n, 2),
        "over_two_sentences": sum(1 for s in sentences if s > 2),
        "missing_cue": sum(1 for t in turns if t["cue"] is None),
        "cues": dict(cues),
        "top_stock_phrase": tic,
        "top_stock_phrase_count": tic_n,
        "volunteered_clock": sum(1 for t in turns if t["clock"]),
        "read_title_verbatim": sum(1 for t in turns if t["title"]),
        "carve_outs": carves,
        "budget": budg,
    }


def _rate(count, total):
    return f"{count}/{total}" + (f"  ({100 * count / total:.0f}%)" if total else "")


def report(r: dict, old: dict | None) -> None:
    n = r["turns"]
    on = old["turns"] if old else n
    oreps = old.get("reps", old.get("carve_outs", {}).get(
        "screen", {}).get("of", 1)) if old else r["reps"]

    def delta(key, value, denom=None, old_denom=None):
        """Compare against a saved run, as a rate when the sample sizes differ.

        A run at --reps 2 against a baseline at --reps 3 was reporting
        "1/2 correct (was 3)", which reads as a collapse and is an improvement.
        Counts are only comparable through their denominators.
        """
        if not old or key not in old:
            return ""
        was = old[key]
        if denom:
            now_pct, was_pct = 100 * value / denom, 100 * was / old_denom
            if round(now_pct) == round(was_pct):
                return "   (unchanged)"
            better = now_pct < was_pct
            return (f"   ({'was' if better else 'WAS'} {was}/{old_denom}"
                    f" = {was_pct:.0f}%)")
        if was == value:
            return "   (unchanged)"
        return f"   ({'was' if value < was else 'WAS'} {was})"

    print(f"\n{'=' * 68}\n  {n} turns over {n // len(SCRIPT)} conversations"
          f"\n{'=' * 68}")

    print("\n  LENGTH        two sentences is the rule")
    print(f"    mean sentences per reply   {r['mean_sentences']}"
          f"{delta('mean_sentences', r['mean_sentences'])}")
    print(f"    replies over two           {_rate(r['over_two_sentences'], n)}"
          f"{delta('over_two_sentences', r['over_two_sentences'], n, on)}")

    print("\n  EXPRESSION    a missing cue leaves the last face on the rig")
    print(f"    replies with no cue        {_rate(r['missing_cue'], n)}"
          f"{delta('missing_cue', r['missing_cue'], n, on)}")
    print(f"    cues used                  {r['cues'] or '{}'}")
    unused = sorted(set(llm_module.EMOTIONS) - set(r["cues"]))
    if unused:
        print(f"    never fired                {', '.join(unused)}")

    print("\n  REPETITION    a stock phrase she reaches for every time is a tic")
    print(f"    most repeated short phrase "
          f"{r['top_stock_phrase'] or '(none)'!r} "
          f"{_rate(r['top_stock_phrase_count'], n)}"
          f"{delta('top_stock_phrase_count', r['top_stock_phrase_count'], n, on)}")

    print("\n  PROHIBITIONS  forbidden in persona.txt and in the situation block")
    print(f"    volunteered the clock      {_rate(r['volunteered_clock'], n)}"
          f"{delta('volunteered_clock', r['volunteered_clock'], n, on)}")
    print(f"    read the title verbatim    {_rate(r['read_title_verbatim'], n)}"
          f"{delta('read_title_verbatim', r['read_title_verbatim'], n, on)}")

    print("\n  CARVE-OUTS    hand-tuned answers, documented in the README")
    for name, c in r["carve_outs"].items():
        was = ""
        if old and name in old.get("carve_outs", {}):
            prev = old["carve_outs"][name]
            now_pct = 100 * c["passed"] / c["of"]
            was_pct = 100 * prev["passed"] / prev["of"]
            was = ("   (unchanged)" if round(now_pct) == round(was_pct)
                   else f"   ({'was' if now_pct > was_pct else 'WAS'} "
                        f"{prev['passed']}/{prev['of']})")
        print(f"    {name:<26} {c['passed']}/{c['of']} correct{was}")

    b = r["budget"]
    print("\n  BUDGET        arithmetic, not sampling")
    print(f"    estimated prompt           {b['estimated_prompt']} of {b['num_ctx']}"
          f", {b['reserved_for_reply']} reserved for the reply")
    print(f"    history kept (long turns)  {b['history_kept']}/{b['history_max']} messages")
    print(f"    {'fits' if b['fits'] else 'DOES NOT FIT -- the persona is what an overflow discards'}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=3, help="conversations to run")
    p.add_argument("--persona", type=Path, help="measure a variant instead")
    p.add_argument("--json", type=Path, help="save this run")
    p.add_argument("--compare", type=Path, help="print deltas against a saved run")
    p.add_argument("--quiet", action="store_true", help="table only")
    args = p.parse_args()

    cfg = Config()
    if args.persona:
        cfg.persona_file = args.persona

    # The clock is pinned for every block that reads it, so two runs differ by
    # sampling alone. `Brain` imported `situation` by name, so this is the one
    # that has to be replaced.
    llm_module.situation = partial(situation, now=NOW)

    print(f"model {cfg.ollama_model}   persona {cfg.persona_file.name}   "
          f"temperature {cfg.temperature}")
    t0 = time.time()
    turns = conversations(cfg, args.reps, verbose=not args.quiet)
    carves = carve_outs(cfg, args.reps)
    result = summarise(turns, carves, budget(cfg))

    old = json.loads(args.compare.read_text()) if args.compare else None
    report(result, old)
    print(f"  {time.time() - t0:.0f}s\n")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"  saved to {args.json}\n")
    return 0 if result["budget"]["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
