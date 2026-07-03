"""Validate the GigaAM-v3 e2e-ctc pipeline before porting it to Rust.

Runs on CI (clean internet). It:
  1. Loads gigaam-v3-e2e-ctc via onnx-asr and prints the encoder ONNX
     input/output names + shapes (the contract the Rust `ort` calls must match).
  2. Synthesizes a known Russian phrase with gTTS (ground truth known).
  3. Transcribes it with onnx-asr (reference).
  4. Transcribes it with a from-scratch reimplementation: our own numpy log-mel
     (a faithful port of onnx-asr's GigaamPreprocessorV3) -> the encoder ONNX ->
     greedy CTC decode. This mirrors exactly what the Rust engine will do.
  5. Compares (3) vs (4) and both vs ground truth.

If (4) matches (3), the Rust port (same feature math + I/O + CTC decode) is
validated.
"""

from __future__ import annotations

import io
import sys

import numpy as np
import onnxruntime as rt
import soundfile as sf

GROUND_TRUTH = "привет как дела сегодня хорошая погода"

# --- GigaAM v3 feature params (from onnx-asr preprocessors/gigaam.py) ---
SAMPLE_RATE = 16_000
N_FFT = SAMPLE_RATE // 50      # 320
WIN_LENGTH = SAMPLE_RATE // 50  # 320
HOP_LENGTH = SAMPLE_RATE // 100  # 160
N_MELS = 64
F_MIN = 0.0
F_MAX = 8_000.0
CLAMP_MIN = 1e-9
CLAMP_MAX = 1e9


def hz_to_mel_htk(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_hz_htk(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)


def melscale_fbanks() -> np.ndarray:
    """[n_freqs=161, n_mels=64] htk mel filterbank (port of fbanks.melscale_fbanks)."""
    n_freqs = N_FFT // 2 + 1
    all_freqs = np.linspace(0, SAMPLE_RATE // 2, n_freqs)
    m_min = hz_to_mel_htk(np.array(F_MIN))
    m_max = hz_to_mel_htk(np.array(F_MAX))
    m_pts = np.linspace(m_min, m_max, N_MELS + 2)
    f_pts = mel_to_hz_htk(m_pts)
    up = (all_freqs[:, None] - f_pts[:-2]) / (f_pts[1:-1] - f_pts[:-2])
    down = (f_pts[2:] - all_freqs[:, None]) / (f_pts[2:] - f_pts[1:-1])
    fb = np.maximum(0.0, np.minimum(up, down))
    return fb.astype(np.float32)


def log_mel_features(waveform: np.ndarray) -> np.ndarray:
    """Port of GigaamPreprocessorV3: STFT(no pad) -> power -> mel -> log(clip).

    Returns [64, T] float32.
    """
    window = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(waveform) - WIN_LENGTH) // HOP_LENGTH
    fb = melscale_fbanks()  # [161, 64]
    feats = np.empty((n_frames, N_MELS), dtype=np.float32)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = waveform[start : start + WIN_LENGTH] * window
        spec = np.fft.rfft(frame, n=N_FFT)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)  # [161]
        mel = power @ fb  # [64]
        feats[i] = np.log(np.clip(mel, CLAMP_MIN, CLAMP_MAX))
    return feats.T  # [64, T]


def ctc_greedy_decode(log_probs: np.ndarray, vocab: list[str], blank: int) -> str:
    ids = log_probs.argmax(axis=-1).tolist()
    out, prev = [], -1
    for t in ids:
        if t != prev and t != blank:
            out.append(vocab[t])
        prev = t
    return "".join(out)


def load_vocab(path: str) -> tuple[list[str], int]:
    vocab: dict[int, str] = {}
    blank = -1
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip("\n"):
                continue
            tok, idx = line.rstrip("\n").rsplit(" ", 1)
            i = int(idx)
            if tok == "<blk>":
                blank = i
                vocab[i] = ""
            else:
                vocab[i] = " " if tok == "" else tok
    size = max(vocab) + 1
    return [vocab.get(i, "") for i in range(size)], blank


def main() -> int:
    import onnx_asr  # noqa: PLC0415
    from gtts import gTTS  # noqa: PLC0415
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    print("=== Loading gigaam-v3-e2e-ctc via onnx-asr ===", flush=True)
    model = onnx_asr.load_model("gigaam-v3-e2e-ctc")

    # Locate the encoder ONNX session inside onnx-asr and print its I/O contract.
    enc = None
    for attr in ("_model", "_encoder"):
        enc = getattr(model, attr, None)
        if isinstance(enc, rt.InferenceSession):
            break
    if isinstance(enc, rt.InferenceSession):
        print("=== Encoder ONNX I/O ===", flush=True)
        for x in enc.get_inputs():
            print(f"  IN  {x.name:20s} {x.shape} {x.type}", flush=True)
        for x in enc.get_outputs():
            print(f"  OUT {x.name:20s} {x.shape} {x.type}", flush=True)

    print(f"\n=== Synthesizing ground truth: '{GROUND_TRUTH}' ===", flush=True)
    buf = io.BytesIO()
    gTTS(GROUND_TRUTH, lang="ru").write_to_fp(buf)
    buf.seek(0)
    import librosa  # noqa: PLC0415

    wav, _ = librosa.load(buf, sr=SAMPLE_RATE, mono=True)
    wav = wav.astype(np.float32)
    sf.write("sample.wav", wav, SAMPLE_RATE)

    print("\n=== Reference (onnx-asr) ===", flush=True)
    ref = model.recognize("sample.wav")
    print(f"  onnx-asr : {ref!r}", flush=True)

    print("\n=== Our reimplementation (numpy log-mel -> encoder ONNX -> CTC) ===", flush=True)
    ctc_path = hf_hub_download("istupakov/gigaam-v3-onnx", "v3_e2e_ctc.onnx")
    vocab_path = hf_hub_download("istupakov/gigaam-v3-onnx", "v3_e2e_ctc_vocab.txt")
    vocab, blank = load_vocab(vocab_path)
    print(f"  vocab size={len(vocab)} blank={blank}", flush=True)

    feats = log_mel_features(wav)[None, :, :]  # [1,64,T]
    feat_len = np.array([feats.shape[2]], dtype=np.int64)
    sess = rt.InferenceSession(ctc_path)
    print("  --- encoder ONNX I/O contract (for the Rust `ort` calls) ---", flush=True)
    for x in sess.get_inputs():
        print(f"    IN  {x.name:20s} {x.shape} {x.type}", flush=True)
    for x in sess.get_outputs():
        print(f"    OUT {x.name:20s} {x.shape} {x.type}", flush=True)
    in_names = [x.name for x in sess.get_inputs()]
    out_names = [x.name for x in sess.get_outputs()]
    feeds = {in_names[0]: feats.astype(np.float32)}
    if len(in_names) > 1:
        feeds[in_names[1]] = feat_len
    log_probs = sess.run(out_names, feeds)[0][0]  # [T, V]
    ours_raw = ctc_greedy_decode(log_probs, vocab, blank)
    ours = ours_raw.replace("▁", " ").strip()  # SentencePiece space -> real space
    print(f"  ours (raw): {ours_raw!r}", flush=True)
    print(f"  ours      : {ours!r}", flush=True)

    print("\n=== Verdict ===", flush=True)
    print(f"  ground truth : {GROUND_TRUTH!r}", flush=True)
    print(f"  onnx-asr     : {str(ref)!r}", flush=True)
    print(f"  ours         : {ours!r}", flush=True)
    norm = lambda s: str(s).lower().replace("!", "").replace("?", "").replace(".", "").replace(",", "").split()
    ok = norm(ours) == norm(ref)
    print(f"  ours ~= onnx-asr (ignoring punct/case): {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
