"""Microphone capture and utterance segmentation."""

import queue
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import FRAME, MIC_RATE, Config
from .vad import StreamingVAD


@dataclass
class Utterance:
    """One complete thing the user said.

    `seq` lets a speculative transcription started mid-silence be matched back
    to the utterance it belongs to, and `speculated` says whether that guess is
    still trustworthy -- speech resuming after the guess was made invalidates it.
    """

    seq: int
    audio: "np.ndarray"
    speculated: bool


class MicListener:
    """Turns a live mic into a queue of complete utterances.

    Runs two threads: sounddevice's audio callback fills a frame queue, and a
    worker drains it through the VAD. Utterances are emitted as float32 arrays
    at 16 kHz. `on_speech_start` fires the moment the gate opens, which is what
    barge-in hangs off -- waiting for the whole utterance would be far too late
    to stop her mid-sentence.
    """

    MAX_SPECULATIONS = 3    # bound the wasted CPU on a rambling utterance

    def __init__(self, cfg: Config, on_speech_start=None, on_speculate=None):
        self.cfg = cfg
        self.on_speech_start = on_speech_start
        self.on_speculate = on_speculate
        self.vad = StreamingVAD(FRAME)
        self.utterances: queue.Queue[Utterance] = queue.Queue()
        self._seq = 0

        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self.muted = threading.Event()   # set => frames are dropped, not analysed
        self._flush = threading.Event()  # set => emit what is buffered, then reset
        # Set while the gate is open: the VAD has heard speech and is still
        # buffering it, so there is an unfinished utterance in flight. Anything
        # that wants to start talking has to check this, not just whether she is
        # already speaking -- see Iris.idle_due.
        self.capturing = threading.Event()
        self._stream = None
        self._worker = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=MIC_RATE,
            blocksize=FRAME,
            dtype="float32",
            channels=1,
            device=self.cfg.input_device,
            callback=self._audio_cb,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self.capturing.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def finish(self) -> None:
        """End the utterance in flight now, and emit it.

        For push-to-talk, where letting go of the key is the end of a turn by
        definition: there is no reason to make somebody sit through the
        trailing-silence timer to prove they have stopped talking.

        This is the opposite of muting, which throws the buffer away -- and it
        has to be, because releasing the key does both at once. The worker
        checks this before it checks `muted`, so the utterance is emitted
        whichever order the two flags were set in.
        """
        self._flush.set()

    def _audio_cb(self, indata, frames, time_info, status):
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # the worker fell behind; dropping is better than blocking audio

    # -- segmentation ------------------------------------------------------
    def _run(self) -> None:
        cfg = self.cfg
        frame_ms = FRAME / MIC_RATE * 1000
        release_frames = max(1, int(cfg.silence_ms / frame_ms))
        # Speculating only pays if it happens well before the endpoint fires.
        speculate_frames = max(1, int(cfg.speculate_ms / frame_ms))
        speculate = self.on_speculate is not None and speculate_frames < release_frames
        min_frames = max(1, int(cfg.min_speech_ms / frame_ms))
        max_frames = int(cfg.max_utterance_s * 1000 / frame_ms)
        preroll = deque(maxlen=max(1, int(cfg.preroll_ms / frame_ms)))

        speaking = False
        silence_run = 0
        spec_count = 0
        spec_valid = False
        buf: list[np.ndarray] = []

        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                frame = None

            # Before the mute check, deliberately. Releasing a push-to-talk key
            # sets both flags, and taken in the other order the mute would
            # discard the very audio the release was asking for.
            if self._flush.is_set():
                self._flush.clear()
                if speaking:
                    speech_frames = len(buf) - silence_run
                    if speech_frames >= min_frames:
                        self.utterances.put(
                            Utterance(self._seq, np.concatenate(buf), spec_valid)
                        )
                speaking, silence_run, buf = False, 0, []
                spec_count, spec_valid = 0, False
                preroll.clear()
                self.capturing.clear()
                continue

            if frame is None:
                continue

            if self.muted.is_set():
                # Half-duplex: discard everything while she talks, and make sure
                # the recurrent state does not carry her voice into the next turn.
                if speaking or buf:
                    speaking, silence_run, buf = False, 0, []
                    spec_count, spec_valid = 0, False
                    self.capturing.clear()
                preroll.clear()
                self.vad.reset()
                continue

            prob = self.vad(frame)

            if not speaking:
                preroll.append(frame)
                if prob >= cfg.vad_threshold:
                    speaking = True
                    silence_run = 0
                    self._seq += 1
                    spec_count, spec_valid = 0, False
                    buf = list(preroll)
                    preroll.clear()
                    # Raised before the callback, so a handler that consults it
                    # sees the utterance that is only just beginning.
                    self.capturing.set()
                    if self.on_speech_start:
                        self.on_speech_start()
            else:
                buf.append(frame)
                if prob < cfg.vad_release:
                    silence_run += 1
                else:
                    spec_valid = False   # she carried on; any guess is now stale
                    silence_run = 0

                # A pause this long is usually the end of a turn. Hand the audio
                # to the transcriber now and let it work through the remaining
                # silence, instead of starting from cold once the endpoint fires.
                if (speculate and silence_run == speculate_frames
                        and spec_count < self.MAX_SPECULATIONS
                        and len(buf) - silence_run >= min_frames):
                    spec_count += 1
                    spec_valid = True
                    self.on_speculate(self._seq, np.concatenate(buf))

                if silence_run >= release_frames or len(buf) >= max_frames:
                    speech_frames = len(buf) - silence_run
                    if speech_frames >= min_frames:
                        self.utterances.put(
                            Utterance(self._seq, np.concatenate(buf), spec_valid)
                        )
                    speaking, silence_run, buf = False, 0, []
                    spec_count, spec_valid = 0, False
                    self.capturing.clear()
