"""Kokoro-82M speech synthesis on the CPU.

The int8 export is the obvious choice on a small machine and is the wrong one
on Zen 3: without VNNI the quantised kernels fall back to slow paths and run
*slower* than fp32. Precision is therefore a config knob, defaulting to fp32.
"""

import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

from .config import TTS_RATE, Config


def _session(path: str, threads: int) -> rt.InferenceSession:
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 4
    sess = rt.InferenceSession(
        path, providers=["CPUExecutionProvider"], sess_options=opts
    )
    sess._model_path = path      # Kokoro.from_session reads this back
    return sess


class Voice:
    def __init__(self, cfg: Config):
        model = cfg.tts_model
        if not model.exists() or not cfg.tts_voices.exists():
            raise FileNotFoundError(
                f"Kokoro weights missing under {model.parent}. "
                "Run: python scripts/fetch_models.py"
            )
        self.cfg = cfg
        self.kokoro = Kokoro.from_session(
            _session(str(model), cfg.tts_threads), str(cfg.tts_voices)
        )

        # Imported lazily: torch is a heavy dependency, and Iris still runs
        # without it installed as long as cloning is off.
        self.cloner = None
        if cfg.clone:
            from .cloning import VoiceCloner
            self.cloner = VoiceCloner(cfg)

        self.warm()

    def warm(self) -> None:
        """First inference pays for graph setup; spend it before she is needed.

        Goes through say() so the conversion pass is warmed too, not just Kokoro.
        """
        self.say("ready when you are")

    def say(self, text: str) -> np.ndarray:
        samples, _rate = self.kokoro.create(
            text, voice=self.cfg.tts_voice, speed=self.cfg.tts_speed, lang="en-us"
        )
        audio = np.asarray(samples, dtype=np.float32)
        if self.cloner is not None:
            # Convert the speech, then pad -- converting silence is wasted work.
            audio = self.cloner.convert(audio)
        pause = int(TTS_RATE * self.cfg.tts_pause_ms / 1000)
        if pause:
            audio = np.concatenate([audio, np.zeros(pause, dtype=np.float32)])
        return audio

    def voices(self) -> list[str]:
        return sorted(self.kokoro.get_voices())
