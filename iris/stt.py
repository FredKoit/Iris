"""Speech to text on the CPU, so the 4 GB of VRAM stays free for the LLM."""

import re

from faster_whisper import WhisperModel

from .config import Config

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

    def __call__(self, audio, hotwords: str | None = None) -> str:
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.whisper_lang,
            beam_size=1,                      # greedy: ~2x faster, fine for speech
            vad_filter=False,                 # already segmented upstream
            condition_on_previous_text=False, # stops runaway repetition loops
            hotwords=hotwords or None,        # names she has learned, see Memory
        )
        parts, weak = [], True
        for seg in segments:
            if seg.no_speech_prob < 0.6 and seg.avg_logprob > -1.0:
                weak = False
            parts.append(seg.text)

        text = re.sub(r"\s+", " ", "".join(parts)).strip()
        if weak or text.lower() in _JUNK or len(text) < 2:
            return ""
        return text
