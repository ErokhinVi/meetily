//! GigaAM-v3 ONNX model wrapper (e2e-ctc and e2e-rnnt).
//!
//! Pipeline (both variants validated against onnx-asr on CI, see
//! scripts/gigaam_validate.py):
//!   waveform 16 kHz mono f32
//!     -> log-mel features [1, 64, T]  (our Rust port of GigaamPreprocessorV3)
//!     -> CTC:  encoder ONNX (`features`,`feature_lengths` -> `log_probs`)
//!              -> greedy CTC decode
//!     -> RNNT: encoder ONNX (`audio_signal`,`length` -> `encoded`,`encoded_len`)
//!              -> greedy transducer loop (decoder + joiner) -> tokens
//!
//! Feature params (GigaAM v3): n_fft = win_length = 320, hop = 160, 64 htk mel
//! bins, f in [0, 8000], log(clip(mel, 1e-9, 1e9)), NO normalization. Space is
//! the SentencePiece marker U+2581 in the vocab (blank = the `<blk>` token).

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::{RealFftPlanner, RealToComplex};

use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::Arc;

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = SAMPLE_RATE / 50; // 320
const WIN_LENGTH: usize = SAMPLE_RATE / 50; // 320
const HOP_LENGTH: usize = SAMPLE_RATE / 100; // 160
const N_MELS: usize = 64;
const N_FREQS: usize = N_FFT / 2 + 1; // 161
const F_MIN: f32 = 0.0;
const F_MAX: f32 = 8_000.0;
const CLAMP_MIN: f32 = 1e-9;
const CLAMP_MAX: f32 = 1e9;
const SENTENCEPIECE_SPACE: char = '\u{2581}';

// RNN-T decoder LSTM hidden size and max emitted tokens per encoder frame.
const PRED_HIDDEN: usize = 320;
const MAX_TOKENS_PER_STEP: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GigaamKind {
    Ctc,
    Rnnt,
}

#[derive(thiserror::Error, Debug)]
pub enum GigaamError {
    #[error("ORT error")]
    Ort(#[from] ort::Error),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Audio too short: need at least {0} samples")]
    AudioTooShort(usize),
}

enum Decode {
    Ctc {
        encoder: Session,
    },
    Rnnt {
        encoder: Session,
        decoder: Session,
        joiner: Session,
    },
}

pub struct GigaamModel {
    decode: Decode,
    vocab: Vec<String>,
    blank_idx: usize,
    hann_window: Array1<f32>,
    /// Mel filterbank, shape [N_FREQS, N_MELS].
    mel_fbank: Array2<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl GigaamModel {
    pub fn new<P: AsRef<Path>>(
        model_dir: P,
        kind: GigaamKind,
        quantized: bool,
    ) -> Result<Self, GigaamError> {
        let dir = model_dir.as_ref();
        let (decode, vocab_file) = match kind {
            GigaamKind::Ctc => (
                Decode::Ctc {
                    encoder: init_session(dir, "v3_e2e_ctc", quantized)?,
                },
                "v3_e2e_ctc_vocab.txt",
            ),
            GigaamKind::Rnnt => (
                Decode::Rnnt {
                    encoder: init_session(dir, "v3_e2e_rnnt_encoder", quantized)?,
                    decoder: init_session(dir, "v3_e2e_rnnt_decoder", quantized)?,
                    joiner: init_session(dir, "v3_e2e_rnnt_joint", quantized)?,
                },
                "v3_e2e_rnnt_vocab.txt",
            ),
        };

        let (vocab, blank_idx) = load_vocab(dir.join(vocab_file))?;
        log::info!(
            "Loaded GigaAM {:?} vocabulary with {} tokens, blank_idx={}",
            kind,
            vocab.len(),
            blank_idx
        );

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);

        Ok(Self {
            decode,
            vocab,
            blank_idx,
            hann_window: hann_window(),
            mel_fbank: melscale_fbanks(),
            fft,
        })
    }

    /// Full transcription of 16 kHz mono f32 samples.
    pub fn transcribe_samples(&mut self, audio: Vec<f32>) -> Result<String, GigaamError> {
        if audio.len() < WIN_LENGTH {
            return Err(GigaamError::AudioTooShort(WIN_LENGTH));
        }

        let features = self.log_mel_features(&audio); // [1, 64, T]
        let n_frames = features.shape()[2] as i64;

        // Borrow `decode` mutably and `vocab`/`blank_idx` immutably — these are
        // disjoint fields, so the borrow checker allows it.
        let vocab = &self.vocab;
        let blank = self.blank_idx;
        match &mut self.decode {
            Decode::Ctc { encoder } => {
                let (flat, t_out, vocab_size) = run_ctc(encoder, &features, n_frames)?;
                Ok(ctc_greedy_decode(&flat, t_out, vocab_size, vocab, blank))
            }
            Decode::Rnnt {
                encoder,
                decoder,
                joiner,
            } => {
                let tokens = run_rnnt(encoder, decoder, joiner, &features, n_frames, blank)?;
                Ok(tokens_to_text(&tokens, vocab))
            }
        }
    }

    /// Port of onnx-asr GigaamPreprocessorV3: STFT (no padding) -> power ->
    /// mel -> log(clip). Returns log-mel features [1, 64, T].
    fn log_mel_features(&self, waveform: &[f32]) -> Array3<f32> {
        let n_frames = 1 + (waveform.len() - WIN_LENGTH) / HOP_LENGTH;
        let mut features = Array3::<f32>::zeros((1, N_MELS, n_frames));

        let mut frame = self.fft.make_input_vec(); // length N_FFT, zero-filled
        let mut spectrum = self.fft.make_output_vec(); // length N_FREQS
        let mut power = [0.0f32; N_FREQS];
        for f in 0..n_frames {
            let start = f * HOP_LENGTH;
            for i in 0..WIN_LENGTH {
                frame[i] = waveform[start + i] * self.hann_window[i];
            }

            self.fft
                .process(&mut frame, &mut spectrum)
                .expect("realfft process: buffer lengths are fixed and correct");
            for (k, cval) in spectrum.iter().enumerate() {
                power[k] = cval.re * cval.re + cval.im * cval.im;
            }

            for m in 0..N_MELS {
                let mut acc = 0.0f32;
                for (k, &p) in power.iter().enumerate() {
                    acc += p * self.mel_fbank[[k, m]];
                }
                features[[0, m, f]] = acc.clamp(CLAMP_MIN, CLAMP_MAX).ln();
            }
        }
        features
    }
}

fn init_session<P: AsRef<Path>>(
    model_dir: P,
    base_name: &str,
    try_quantized: bool,
) -> Result<Session, GigaamError> {
    let providers = vec![CPUExecutionProvider::default().build()];

    let quantized = model_dir.as_ref().join(format!("{}.int8.onnx", base_name));
    let filename = if try_quantized && quantized.exists() {
        format!("{}.int8.onnx", base_name)
    } else {
        format!("{}.onnx", base_name)
    };
    log::info!("Loading GigaAM ONNX: {}", filename);

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers(providers)?
        .with_parallel_execution(true)?
        .commit_from_file(model_dir.as_ref().join(filename))?;

    Ok(session)
}

/// Vocab file lines are "<token> <id>"; blank token is "<blk>".
fn load_vocab<P: AsRef<Path>>(vocab_path: P) -> Result<(Vec<String>, usize), GigaamError> {
    let content = fs::read_to_string(vocab_path)?;

    let mut tokens: Vec<(String, usize)> = Vec::new();
    let mut blank_idx: Option<usize> = None;
    let mut max_id = 0usize;

    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        // Token may itself be empty or contain spaces; split on the LAST space.
        let Some((token, id_str)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(id) = id_str.trim().parse::<usize>() else {
            continue;
        };
        if token == "<blk>" {
            blank_idx = Some(id);
        }
        tokens.push((token.to_string(), id));
        max_id = max_id.max(id);
    }

    let mut vocab = vec![String::new(); max_id + 1];
    for (token, id) in tokens {
        vocab[id] = token.replace(SENTENCEPIECE_SPACE, " ");
    }

    let blank_idx = blank_idx.ok_or_else(|| {
        GigaamError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Missing <blk> token in GigaAM vocabulary",
        ))
    })?;

    Ok((vocab, blank_idx))
}

/// CTC encoder: features [1,64,T] -> log_probs [1, T', V]. Returns (flat, T', V).
fn run_ctc(
    encoder: &mut Session,
    features: &Array3<f32>,
    n_frames: i64,
) -> Result<(Vec<f32>, usize, usize), GigaamError> {
    let feature_lengths = Array1::from_vec(vec![n_frames]);
    let inputs = inputs![
        "features" => TensorRef::from_array_view(features.view())?,
        "feature_lengths" => TensorRef::from_array_view(feature_lengths.view())?,
    ];
    let outputs = encoder.run(inputs)?;
    let log_probs = outputs
        .get("log_probs")
        .ok_or_else(|| GigaamError::OutputNotFound("log_probs".to_string()))?
        .try_extract_array::<f32>()?;
    let shape = log_probs.shape();
    let (t_out, vocab_size) = (shape[1], shape[2]);
    let flat: Vec<f32> = log_probs.iter().copied().collect();
    Ok((flat, t_out, vocab_size))
}

/// Greedy CTC: argmax per frame, collapse repeats, drop blank, map to vocab.
fn ctc_greedy_decode(
    log_probs: &[f32],
    t_out: usize,
    vocab_size: usize,
    vocab: &[String],
    blank_idx: usize,
) -> String {
    let mut text = String::new();
    let mut prev = usize::MAX;
    for t in 0..t_out {
        let row = &log_probs[t * vocab_size..(t + 1) * vocab_size];
        let best = argmax(row);
        if best != prev && best != blank_idx {
            if let Some(tok) = vocab.get(best) {
                text.push_str(tok);
            }
        }
        prev = best;
    }
    text.trim().to_string()
}

/// Greedy RNN-T (port of onnx-asr `_AsrWithTransducerDecoding._decoding`).
fn run_rnnt(
    encoder: &mut Session,
    decoder: &mut Session,
    joiner: &mut Session,
    features: &Array3<f32>,
    n_frames: i64,
    blank_idx: usize,
) -> Result<Vec<usize>, GigaamError> {
    // Encoder: audio_signal [1,64,T] + length -> encoded [1,768,T'] + encoded_len.
    let length = Array1::from_vec(vec![n_frames]);
    let (encoded, enc_len) = {
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(features.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = encoder.run(inputs)?;
        let encoded = outputs
            .get("encoded")
            .ok_or_else(|| GigaamError::OutputNotFound("encoded".to_string()))?
            .try_extract_array::<f32>()?
            .to_owned();
        let enc_len = outputs
            .get("encoded_len")
            .ok_or_else(|| GigaamError::OutputNotFound("encoded_len".to_string()))?
            .try_extract_array::<i32>()?;
        let n = enc_len.iter().next().copied().unwrap_or(0) as usize;
        (encoded, n)
    };

    let d_model = encoded.shape()[1]; // 768
    let n = enc_len.min(encoded.shape()[2]);

    // LSTM state (h, c) and cached decoder output.
    let mut h: ArrayD<f32> = ArrayD::zeros(IxDyn(&[1, 1, PRED_HIDDEN]));
    let mut c: ArrayD<f32> = ArrayD::zeros(IxDyn(&[1, 1, PRED_HIDDEN]));
    let mut dec_out: Option<ArrayD<f32>> = None; // None => decoder must run
    let mut pending_h = h.clone();
    let mut pending_c = c.clone();

    let mut tokens: Vec<usize> = Vec::new();
    let mut t = 0usize;
    let mut emitted = 0usize;

    while t < n {
        if dec_out.is_none() {
            let prev = tokens.last().copied().unwrap_or(blank_idx) as i64;
            let x = Array2::from_shape_vec((1, 1), vec![prev])?;
            let inputs = inputs![
                "x" => TensorRef::from_array_view(x.view())?,
                "h.1" => TensorRef::from_array_view(h.view())?,
                "c.1" => TensorRef::from_array_view(c.view())?,
            ];
            let outputs = decoder.run(inputs)?;
            dec_out = Some(
                outputs
                    .get("dec")
                    .ok_or_else(|| GigaamError::OutputNotFound("dec".to_string()))?
                    .try_extract_array::<f32>()?
                    .to_owned(),
            );
            pending_h = outputs
                .get("h")
                .ok_or_else(|| GigaamError::OutputNotFound("h".to_string()))?
                .try_extract_array::<f32>()?
                .to_owned();
            pending_c = outputs
                .get("c")
                .ok_or_else(|| GigaamError::OutputNotFound("c".to_string()))?
                .try_extract_array::<f32>()?
                .to_owned();
        }

        // joiner: enc = encoded[0,:,t] as [1,768,1]; dec = dec_out^T as [1,320,1].
        // Transposing [1,1,320] -> [1,320,1] keeps the flat element order, so we
        // just rebuild the tensor with the reshaped dims.
        let enc_col: Vec<f32> = encoded
            .index_axis(Axis(0), 0)
            .index_axis(Axis(1), t)
            .iter()
            .copied()
            .collect();
        let enc_in = Array3::from_shape_vec((1, d_model, 1), enc_col)?;
        let dec_ref = dec_out.as_ref().expect("dec_out set above");
        let dec_flat: Vec<f32> = dec_ref.iter().copied().collect();
        let dec_in = Array3::from_shape_vec((1, dec_flat.len(), 1), dec_flat)?;

        let token = {
            let inputs = inputs![
                "enc" => TensorRef::from_array_view(enc_in.view())?,
                "dec" => TensorRef::from_array_view(dec_in.view())?,
            ];
            let outputs = joiner.run(inputs)?;
            let joint = outputs
                .get("joint")
                .ok_or_else(|| GigaamError::OutputNotFound("joint".to_string()))?
                .try_extract_array::<f32>()?;
            let flat: Vec<f32> = joint.iter().copied().collect(); // [1,1,1,V] -> V
            argmax(&flat)
        };

        if token != blank_idx {
            h = pending_h.clone();
            c = pending_c.clone();
            dec_out = None;
            tokens.push(token);
            emitted += 1;
        }
        if token == blank_idx || emitted == MAX_TOKENS_PER_STEP {
            t += 1;
            emitted = 0;
        }
    }

    Ok(tokens)
}

fn tokens_to_text(tokens: &[usize], vocab: &[String]) -> String {
    let mut text = String::new();
    for &t in tokens {
        if let Some(tok) = vocab.get(t) {
            text.push_str(tok);
        }
    }
    text.trim().to_string()
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

fn hann_window() -> Array1<f32> {
    // np.hanning(WIN_LENGTH + 1)[:-1]
    let n = WIN_LENGTH;
    let mut w = Array1::<f32>::zeros(n);
    for i in 0..n {
        w[i] = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
    }
    w
}

fn hz_to_mel_htk(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// htk mel filterbank, shape [N_FREQS, N_MELS] (port of fbanks.melscale_fbanks).
fn melscale_fbanks() -> Array2<f32> {
    let mut all_freqs = vec![0.0f32; N_FREQS];
    let nyquist = (SAMPLE_RATE / 2) as f32;
    for (k, freq) in all_freqs.iter_mut().enumerate() {
        *freq = nyquist * k as f32 / (N_FREQS - 1) as f32;
    }

    let m_min = hz_to_mel_htk(F_MIN);
    let m_max = hz_to_mel_htk(F_MAX);
    let mut f_pts = vec![0.0f32; N_MELS + 2];
    for (i, pt) in f_pts.iter_mut().enumerate() {
        let mel = m_min + (m_max - m_min) * i as f32 / (N_MELS + 1) as f32;
        *pt = mel_to_hz_htk(mel);
    }

    let mut fb = Array2::<f32>::zeros((N_FREQS, N_MELS));
    for k in 0..N_FREQS {
        for m in 0..N_MELS {
            let up = (all_freqs[k] - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
            let down = (f_pts[m + 2] - all_freqs[k]) / (f_pts[m + 2] - f_pts[m + 1]);
            fb[[k, m]] = up.min(down).max(0.0);
        }
    }
    fb
}
