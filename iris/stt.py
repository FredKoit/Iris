"""Speech to text on the CPU, so the 4 GB of VRAM stays free for the LLM.

Two bars, not one. Whisper says how sure it is and that number used to be
collapsed into "is this speech at all" and thrown away -- which is the right
question for whether to answer someone and the wrong one for whether to believe
them permanently. Answering a half-heard sentence costs one bad reply; writing
it into memory costs every session after it. `Heard.clear` is the higher bar,
and only memory reads it.
"""

import re
from typing import NamedTuple

from faster_whisper import WhisperModel

from .config import Config


class Heard(NamedTuple):
    text: str
    logprob: float     # mean over segments; roughly -0.2 clean, under -0.6 mumbled
    clear: bool        # confident enough to be remembered, not merely answered

# Whisper reliably hallucinates these over near-silence. Dropping them is the
# difference between a companion and one that talks to itself all evening.
_JUNK = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "you", "bye.", "bye bye.", ".", "so", "okay.", "oh.", "hmm.",
    "thanks for watching.", "please subscribe.", "subscribe!",
}


class Transcriber:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.whisper_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=cfg.whisper_threads,
        )

    def warm(self) -> None:
        """The first transcription allocates its workspace; do it up front."""
        import numpy as np
        self(np.zeros(8000, dtype=np.float32))

    def __call__(self, audio, hotwords: str | None = None) -> Heard:
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.whisper_lang,
            beam_size=1,                      # greedy: ~2x faster, fine for speech
            vad_filter=False,                 # already segmented upstream
            condition_on_previous_text=False, # stops runaway repetition loops
            hotwords=hotwords or None,        # names she has learned, see Memory
        )
        parts, weak, scores = [], True, []
        for seg in segments:
            if seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0:
                weak = False
            # Every segment counts towards the memory bar, including the ones
            # that failed the bar above: a sentence that is half clean speech
            # and half mumbling is exactly the kind that should be answered and
            # not believed.
            scores.append((seg.avg_logprob, seg.no_speech_prob))
            parts.append(seg.text)

        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        if weak or text.lower() in _JUNK or len(text) < 2:
            return Heard("", 0.0, False)

        logprob = sum(s for s, _ in scores) / len(scores)
        clear = (
            logprob >= self.cfg.stt_clear_logprob
            and max(p for _, p in scores) <= self.cfg.stt_clear_no_speech
        )
        return Heard(text, logprob, clear)
