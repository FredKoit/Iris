"""Re-voice Kokoro's output with a cloned timbre.

Kokoro cannot be given a new voice: its voicepacks are style vectors from a
training pipeline that was never released, and the only "custom voice" it
supports natively is a blend of the 54 it ships with. So the timbre is changed
*after* synthesis instead, by the Kanade zero-shot voice-conversion model --
Kokoro provides the words and the prosody, the reference recording provides the
voice.

The cost is a second neural pass over every phrase. Two things keep that
affordable:

- The reference is encoded **once**. `KanadeModel.voice_conversion()` re-encodes
  it on every call, which on a 9.8 s reference is 1.12 s of pointless work per
  phrase. It is only a wrapper over encode+decode, so the reference's global
  embedding is computed at startup and reused. Measured on the opening phrase:
  1.59 s to 0.48 s.
- Conversion runs at RTF 0.29, and Kokoro at 0.43. Together that is 0.72, still
  under 1.0, so she does not fall behind mid-reply once she has started talking.

Note that the decoder is stochastic: the same sentence converts slightly
differently every time. Running identical code twice gives waveforms differing
by up to 0.68 peak. That is the model, not a bug here.
"""

import numpy as np
import torch

from .config import TTS_RATE, Config

# The mel decoder's RoPE table is precomputed for 1024 frames at hop 256, so a
# single pass tops out just under 11 s of audio. Her phrases are 1-4 s, but a
# long unbroken tail from the chunker could exceed it.
_MAX_SAMPLES = (1024 - 1) * 256          # 261,888 samples, 10.9 s at 24 kHz
_SAFE_SAMPLES = int(_MAX_SAMPLES * 0.75)

# Below roughly a syllable there is nothing for the encoder to work with, and a
# stray fragment is not worth 0.3 s.
_MIN_SAMPLES = 2400                      # 0.1 s


class VoiceCloner:
    def __init__(self, cfg: Config):
        from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode

        if not cfg.clone_reference.exists():
            raise FileNotFoundError(
                f"reference recording not found: {cfg.clone_reference}. "
                "Point clone_reference at a 3-10 second clip of the target voice."
            )

        self._vocode = vocode
        torch.set_num_threads(cfg.clone_threads)
        self.device = torch.device("cpu")

        self.model = KanadeModel.from_pretrained(cfg.clone_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.model.config.vocoder_name).to(self.device)
        self.rate = self.model.config.sample_rate
        if self.rate != TTS_RATE:
            raise RuntimeError(
                f"conversion runs at {self.rate} Hz but Kokoro emits {TTS_RATE} Hz; "
                "resampling is not implemented"
            )

        reference = load_audio(str(cfg.clone_reference), sample_rate=self.rate)
        with torch.inference_mode():
            self.reference = self.model.encode(
                reference.to(self.device), return_content=False, return_global=True
            ).global_embedding
        self.reference_seconds = reference.shape[-1] / self.rate
        self.failures = 0

    # -- conversion --------------------------------------------------------
    def _convert_one(self, audio: np.ndarray) -> np.ndarray:
        source = torch.from_numpy(np.ascontiguousarray(audio)).to(self.device)
        with torch.inference_mode():
            content = self.model.encode(
                source, return_content=True, return_global=False
            ).content_embedding
            mel = self.model.decode(
                content_embedding=content,
                global_embedding=self.reference,
                target_audio_length=source.size(0),
            )
            wav = self._vocode(self.vocoder, mel.unsqueeze(0))
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def convert(self, audio: np.ndarray) -> np.ndarray:
        """Re-voice one phrase. Returns the original audio if conversion fails.

        A dropped phrase would be a hole in the middle of a sentence, and a
        crash would end the conversation; her own voice is a better failure.
        """
        if audio.size < _MIN_SAMPLES:
            return audio
        try:
            if audio.size <= _SAFE_SAMPLES:
                return self._convert_one(audio)
            pieces = [
                self._convert_one(audio[i:i + _SAFE_SAMPLES])
                for i in range(0, audio.size, _SAFE_SAMPLES)
            ]
            return np.concatenate(pieces)
        except Exception as exc:
            self.failures += 1
            if self.failures <= 3:
                print(f"  [voice conversion failed, using her own voice: {exc}]",
                      flush=True)
            return audio
