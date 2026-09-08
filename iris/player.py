"""Output audio with instant barge-in."""

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from .config import TTS_RATE, Config


class Player:
    """A callback-driven output stream you can silence in one buffer period.

    Clips are queued whole; the callback walks them. `stop()` drops everything
    still queued, which is what makes interrupting her feel immediate rather
    than "she finishes the sentence first".

    A clip can carry a `cue` -- an emotion tag belonging to the phrase in it.
    Synthesis runs ahead of playback, so anything acted on at queue time happens
    while an earlier phrase is still coming out of the speakers. The cue is
    published on `cues` at the moment the clip's first sample is actually sent
    to the device, which is the moment the phrase is heard.
    """

    def __init__(self, cfg: Config):
        self._clips: deque[tuple[np.ndarray, object]] = deque()
        self._pos = 0
        # Read by whoever drives the avatar. A queue rather than a callback
        # because the audio callback must not be made to wait on a websocket.
        self.cues: queue.Queue = queue.Queue()
        # RMS of the block just sent to the speakers. The avatar's mouth follows
        # this, which is why lip sync needs no virtual audio cable.
        self.level = 0.0
        self._lock = threading.Lock()
        self.drained = threading.Event()
        self.drained.set()
        self._stream = sd.OutputStream(
            samplerate=TTS_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=cfg.output_device,
            callback=self._cb,
        )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def play(self, samples: np.ndarray, cue: object = None) -> None:
        with self._lock:
            self._clips.append((samples.astype(np.float32), cue))
            self.drained.clear()

    def stop(self) -> None:
        with self._lock:
            self._clips.clear()
            self._pos = 0
            self.drained.set()
        # Cues for phrases that will now never be heard. Dropping them keeps a
        # stale expression from landing on her face after she was cut off.
        while True:
            try:
                self.cues.get_nowait()
            except queue.Empty:
                break

    @property
    def busy(self) -> bool:
        return not self.drained.is_set()

    def _cb(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            while filled < frames and self._clips:
                clip, cue = self._clips[0]
                if self._pos == 0 and cue is not None:
                    # First sample of this clip is going out now. put_nowait on
                    # an unbounded queue does not block, so the device callback
                    # never waits on the consumer.
                    self.cues.put_nowait(cue)
                take = min(frames - filled, len(clip) - self._pos)
                out[filled:filled + take] = clip[self._pos:self._pos + take]
                filled += take
                self._pos += take
                if self._pos >= len(clip):
                    self._clips.popleft()
                    self._pos = 0
            if not self._clips:
                self.drained.set()
        if filled < frames:
            out[filled:] = 0.0
        self.level = float(np.sqrt(np.mean(out[:frames] ** 2))) if filled else 0.0
