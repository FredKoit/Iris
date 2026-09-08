"""Inspect and edit what she remembers.

  python scripts/memory.py              what she knows, strongest first
  python scripts/memory.py log 20       the last 20 things said
  python scripts/memory.py forget 3     delete fact 3
  python scripts/memory.py wipe         start over
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris.config import Config
from iris.memory import Memory


def main() -> int:
    cfg = Config()
    if not cfg.memory_db.exists():
        print(f"no memory yet at {cfg.memory_db}")
        return 0
    mem = Memory(cfg)
    args = sys.argv[1:]
    cmd = args[0] if args else "list"

    if cmd == "list":
        st = mem.stats()
        print(f"{st['facts']} facts, {st['turns']} turns, "
              f"{st['pending']} awaiting consolidation\n")
        for fid, text, hits, quote, score in mem.detail(limit=999):
            seen = f"x{hits}" if hits > 1 else "  "
            # The score is what decides the order, and what decides which facts
            # fit the prompt budget below. A fact well down this list is one she
            # has stopped being reminded of.
            print(f"  {fid:>3}  {seen} {score:5.2f}  {text}")
            # The words it was drawn from, so a fact you do not recognise can
            # be traced back instead of just deleted on suspicion.
            if quote:
                print(f"                  from: \"{quote}\"")
        budget = cfg.memory_char_budget
        used = len(mem.profile())
        print(f"\nprompt block: {used}/{budget} chars (~{used // 4} tokens)")

    elif cmd == "log":
        n = int(args[1]) if len(args) > 1 else 20
        rows = mem.db.execute(
            "SELECT role, text FROM turns ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        for role, text in reversed(rows):
            who = "you " if role == "user" else "iris"
            print(f"{who}: {text}")

    elif cmd == "forget":
        if len(args) < 2:
            print("which one? run with no arguments to see the ids")
            return 1
        mem.forget(int(args[1]))
        print(f"forgot {args[1]}")

    elif cmd == "wipe":
        confirm = input(f"delete {cfg.memory_db}? [y/N] ").strip().lower()
        if confirm == "y":
            mem.close()
            cfg.memory_db.unlink()
            print("wiped")
            return 0
        print("kept")

    else:
        print(__doc__)
        return 1

    mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
