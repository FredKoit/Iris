"""Streaming voice-activity detection.

faster-whisper ships Silero VAD v6 as an ONNX asset, so we reuse that file
instead of pulling in torch. Its own wrapper is batch-oriented and resets the
recurrent state on every call; for a live mic we need the state carried across
frames, so this is a thin stateful re-implementation over the same graph.
"""

import numpy as np
import onnxruntime
from faster_whisper.utils import get_assets_path
from pathlib import Path

CONTEXT = 64          # samples of previous frame the model conditions on
STATE = 128


class StreamingVAD:
    def __init__(self, frame: int = 512):
        path = Path(get_assets_path()) / "silero_vad_v6.onnx"
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.frame = frame
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, STATE), dtype=np.float32)
        self._c = np.zeros((1, 1, STATE), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample float32 frame."""
        if frame.shape[0] != self.frame:
            raise ValueError(f"expected {self.frame} samples, got {frame.shape[0]}")
        x = frame.reshape(1, -1).astype(np.float32)
        inp = np.concatenate([self._context, x], axis=1)
        out, self._h, self._c = self.session.run(
            None, {"input": inp, "h": self._h, "c": self._c}
        )
        self._context = x[:, -CONTEXT:]
        return float(np.asarray(out).ravel()[0])
