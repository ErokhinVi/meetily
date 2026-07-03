//! GigaAM-v3 e2e-ctc ONNX model wrapper.
//!
//! Pipeline (validated against onnx-asr on CI, see scripts/gigaam_validate.py):
//!   waveform 16 kHz mono f32
//!     -> log-mel features [1, 64, T]  (our Rust port of GigaamPreprocessorV3)
//!     -> encoder ONNX (inputs `features`, `feature_lengths`; output `log_probs`)
//!     -> greedy CTC decode -> text
//!
//! Feature params (GigaAM v3): n_fft = win_length = 320, hop = 160, 64 htk mel
//! bins, f in [0, 8000], log(clip(mel, 1e-9, 1e9)), NO normalization. Space is
//! the SentencePiece marker U+2581 in the 257-token vocab (blank = last index).

use ndarray::{Array1, Array2, Array3};
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

pub struct GigaamModel {
    encoder: Session,
    vocab: Vec<String>,
    blank_idx: usize,
    hann_window: Array1<f32>,
    /// Mel filterbank, shape [N_FREQS, N_MELS].
    mel_fbank: Array2<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl GigaamModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<Self, GigaamError> {
        let encoder = Self::init_session(&model_dir, quantized)?;
        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;

        log::info!(
            "Loaded GigaAM vocabulary with {} tokens, blank_idx={}",
            vocab.len(),
            blank_idx
        );

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);

        Ok(Self {
            encoder,
            vocab,
            blank_idx,
            hann_window: hann_window(),
            mel_fbank: melscale_fbanks(),
            fft,
        })
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        try_quantized: bool,
    ) -> Result<Session, GigaamError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        let filename = {
            let quantized = model_dir.as_ref().join("v3_e2e_ctc.int8.onnx");
            if try_quantized && quantized.exists() {
                log::info!("Loading quantized GigaAM model (v3_e2e_ctc.int8.onnx)");
                "v3_e2e_ctc.int8.onnx"
            } else {
                log::info!("Loading GigaAM model (v3_e2e_ctc.onnx)");
                "v3_e2e_ctc.onnx"
            }
        };

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(model_dir.as_ref().join(filename))?;

        Ok(session)
    }

    /// Vocab file lines are "<token> <id>"; blank token is "<blk>".
    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, usize), GigaamError> {
        let vocab_path = model_dir.as_ref().join("v3_e2e_ctc_vocab.txt");
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

    /// Full transcription of 16 kHz mono f32 samples.
    pub fn transcribe_samples(&mut self, audio: Vec<f32>) -> Result<String, GigaamError> {
        if audio.len() < WIN_LENGTH {
            return Err(GigaamError::AudioTooShort(WIN_LENGTH));
        }

        let features = self.log_mel_features(&audio); // [1, 64, T]
        let n_frames = features.shape()[2] as i64;
        let feature_lengths = Array1::from_vec(vec![n_frames]);

        let inputs = inputs![
            "features" => TensorRef::from_array_view(features.view())?,
            "feature_lengths" => TensorRef::from_array_view(feature_lengths.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let log_probs = outputs
            .get("log_probs")
            .ok_or_else(|| GigaamError::OutputNotFound("log_probs".to_string()))?
            .try_extract_array::<f32>()?;

        // log_probs: [1, T', V]
        let shape = log_probs.shape();
        let (t_out, vocab_size) = (shape[1], shape[2]);
        let flat = log_probs
            .as_slice()
            .ok_or_else(|| GigaamError::OutputNotFound("log_probs contiguous".to_string()))?;

        Ok(self.ctc_greedy_decode(flat, t_out, vocab_size))
    }

    /// Greedy CTC: argmax per frame, collapse repeats, drop blank, map to vocab.
    fn ctc_greedy_decode(&self, log_probs: &[f32], t_out: usize, vocab_size: usize) -> String {
        let mut text = String::new();
        let mut prev = usize::MAX;
        for t in 0..t_out {
            let row = &log_probs[t * vocab_size..(t + 1) * vocab_size];
            let mut best = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            if best != prev && best != self.blank_idx {
                if let Some(tok) = self.vocab.get(best) {
                    text.push_str(tok);
                }
            }
            prev = best;
        }
        text.trim().to_string()
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
            // Windowed frame (win_length == n_fft, so no zero padding needed).
            for i in 0..WIN_LENGTH {
                frame[i] = waveform[start + i] * self.hann_window[i];
            }

            // Real FFT power spectrum: |X[k]|^2 for k in 0..=N_FFT/2.
            self.fft
                .process(&mut frame, &mut spectrum)
                .expect("realfft process: buffer lengths are fixed and correct");
            for (k, c) in spectrum.iter().enumerate() {
                power[k] = c.re * c.re + c.im * c.im;
            }

            // Mel: power [161] x fbank [161, 64] -> [64], then log(clip).
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
