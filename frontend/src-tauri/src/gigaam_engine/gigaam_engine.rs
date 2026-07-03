//! GigaAM-v3 engine: model registry, download, load, transcribe.
//!
//! Mirrors the Parakeet engine shape but is simpler: each model is a single
//! encoder ONNX plus a vocab file downloaded from the public HF repo
//! `istupakov/gigaam-v3-onnx`. GigaAM-v3 is Russian-only.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::sync::RwLock;

use crate::gigaam_engine::model::{GigaamKind, GigaamModel};

const HF_BASE: &str = "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading,
    Corrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size_mb: u32,
    pub description: String,
    pub language: String,
    pub status: ModelStatus,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub percent: u32,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub downloaded_mb: f64,
    pub total_mb: f64,
    pub speed_mbps: f64,
}

pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// One registry entry: (model name, kind, [(remote filename, local filename, size bytes)], size_mb, description).
struct ModelSpec {
    name: &'static str,
    kind: GigaamKind,
    files: &'static [(&'static str, &'static str, u64)],
    size_mb: u32,
    description: &'static str,
}

// int8 quantized, both ~225 MB. CTC is faster; RNN-T is the more accurate variant.
const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        name: "gigaam-v3-e2e-ctc-int8",
        kind: GigaamKind::Ctc,
        files: &[
            ("v3_e2e_ctc.int8.onnx", "v3_e2e_ctc.int8.onnx", 224_891_000),
            ("v3_e2e_ctc_vocab.txt", "v3_e2e_ctc_vocab.txt", 3_000),
        ],
        size_mb: 225,
        description: "GigaAM-v3 CTC (Сбер) — русский, быстрый, с пунктуацией",
    },
    ModelSpec {
        name: "gigaam-v3-e2e-rnnt-int8",
        kind: GigaamKind::Rnnt,
        files: &[
            ("v3_e2e_rnnt_encoder.int8.onnx", "v3_e2e_rnnt_encoder.int8.onnx", 224_570_000),
            ("v3_e2e_rnnt_decoder.int8.onnx", "v3_e2e_rnnt_decoder.int8.onnx", 1_160_000),
            ("v3_e2e_rnnt_joint.int8.onnx", "v3_e2e_rnnt_joint.int8.onnx", 690_000),
            ("v3_e2e_rnnt_vocab.txt", "v3_e2e_rnnt_vocab.txt", 12_000),
        ],
        size_mb: 226,
        description: "GigaAM-v3 RNN-T (Сбер) — русский, максимальная точность, с пунктуацией",
    },
];

fn spec(name: &str) -> Option<&'static ModelSpec> {
    REGISTRY.iter().find(|m| m.name == name)
}

pub struct GigaamEngine {
    models_dir: PathBuf, // app_data/models/gigaam
    model: RwLock<Option<GigaamModel>>,
    current_model: RwLock<Option<String>>,
    pub available_models: RwLock<HashMap<String, ModelStatus>>,
    pub active_downloads: RwLock<HashSet<String>>,
}

impl GigaamEngine {
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let base = models_dir.ok_or_else(|| anyhow!("models directory not set"))?;
        let dir = base.join("gigaam");
        std::fs::create_dir_all(&dir)?;
        Ok(Self {
            models_dir: dir,
            model: RwLock::new(None),
            current_model: RwLock::new(None),
            available_models: RwLock::new(HashMap::new()),
            active_downloads: RwLock::new(HashSet::new()),
        })
    }

    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    fn model_dir(&self, name: &str) -> PathBuf {
        self.models_dir.join(name)
    }

    /// A model is Available when all its files exist on disk with a plausible size.
    fn disk_status(&self, s: &ModelSpec) -> ModelStatus {
        let dir = self.model_dir(s.name);
        let mut any_present = false;
        for (_, local, expected) in s.files {
            let path = dir.join(local);
            match std::fs::metadata(&path) {
                Ok(meta) => {
                    any_present = true;
                    // Allow slack; treat clearly-truncated files as corrupted.
                    if meta.len() < expected / 2 {
                        return ModelStatus::Corrupted;
                    }
                }
                Err(_) => {
                    if any_present {
                        return ModelStatus::Corrupted;
                    }
                    return ModelStatus::Missing;
                }
            }
        }
        ModelStatus::Available
    }

    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        let downloading = self.active_downloads.read().await.clone();
        let mut out = Vec::new();
        let mut status_map = HashMap::new();
        for s in REGISTRY {
            let status = if downloading.contains(s.name) {
                ModelStatus::Downloading
            } else {
                self.disk_status(s)
            };
            status_map.insert(s.name.to_string(), status);
            out.push(ModelInfo {
                name: s.name.to_string(),
                size_mb: s.size_mb,
                description: s.description.to_string(),
                language: "ru".to_string(),
                status,
            });
        }
        *self.available_models.write().await = status_map;
        Ok(out)
    }

    pub async fn is_model_loaded(&self) -> bool {
        self.model.read().await.is_some()
    }

    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model.read().await.clone()
    }

    pub async fn load_model(&self, name: &str) -> Result<()> {
        let s = spec(name).ok_or_else(|| anyhow!("Unknown GigaAM model: {}", name))?;
        if matches!(self.disk_status(s), ModelStatus::Missing | ModelStatus::Corrupted) {
            return Err(anyhow!("GigaAM model '{}' is not fully downloaded", name));
        }
        let dir = self.model_dir(name);
        let kind = s.kind;
        // GigaamModel::new is CPU/IO heavy; run it off the async executor.
        let model = tokio::task::spawn_blocking(move || GigaamModel::new(&dir, kind, true))
            .await
            .map_err(|e| anyhow!("load task join error: {}", e))?
            .map_err(|e| anyhow!("Failed to load GigaAM model: {}", e))?;

        *self.model.write().await = Some(model);
        *self.current_model.write().await = Some(name.to_string());
        log::info!("GigaAM model loaded: {}", name);
        Ok(())
    }

    pub async fn transcribe_audio(&self, audio_data: Vec<f32>) -> Result<String> {
        let mut guard = self.model.write().await;
        let model = guard
            .as_mut()
            .ok_or_else(|| anyhow!("No GigaAM model loaded"))?;
        let duration = audio_data.len() as f64 / 16_000.0;
        log::debug!("GigaAM transcribing {} samples ({:.1}s)", audio_data.len(), duration);
        model
            .transcribe_samples(audio_data)
            .map_err(|e| anyhow!("GigaAM transcription failed: {}", e))
    }

    pub async fn cancel_download(&self, name: &str) -> Result<()> {
        self.active_downloads.write().await.remove(name);
        Ok(())
    }

    pub async fn delete_model(&self, name: &str) -> Result<String> {
        // Unload if it is the current model.
        if self.current_model.read().await.as_deref() == Some(name) {
            *self.model.write().await = None;
            *self.current_model.write().await = None;
        }
        let dir = self.model_dir(name);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        self.discover_models().await?;
        Ok(format!("Deleted GigaAM model {}", name))
    }

    pub async fn download_model_detailed(
        &self,
        name: &str,
        progress: Option<ProgressCallback>,
    ) -> Result<()> {
        let s = spec(name).ok_or_else(|| anyhow!("Unknown GigaAM model: {}", name))?;
        self.active_downloads.write().await.insert(name.to_string());

        let result = self.download_files(s, progress).await;

        self.active_downloads.write().await.remove(name);
        self.discover_models().await.ok();
        result
    }

    async fn download_files(&self, s: &ModelSpec, progress: Option<ProgressCallback>) -> Result<()> {
        let dir = self.model_dir(s.name);
        std::fs::create_dir_all(&dir)?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .build()?;

        let total_bytes: u64 = s.files.iter().map(|(_, _, sz)| *sz).sum();
        let mut done_bytes: u64 = 0;
        let start = Instant::now();

        for (remote, local, _) in s.files {
            let url = format!("{}/{}", HF_BASE, remote);
            let dest = dir.join(local);
            log::info!("Downloading GigaAM file {} -> {}", url, dest.display());

            let resp = client.get(&url).send().await?.error_for_status()?;
            let mut file = tokio::fs::File::create(&dest).await?;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                if !self.active_downloads.read().await.contains(s.name) {
                    return Err(anyhow!("Download cancelled"));
                }
                let chunk = chunk?;
                file.write_all(&chunk).await?;
                done_bytes += chunk.len() as u64;

                if let Some(cb) = &progress {
                    let elapsed = start.elapsed().as_secs_f64().max(0.001);
                    let downloaded_mb = done_bytes as f64 / 1_048_576.0;
                    cb(DownloadProgress {
                        percent: ((done_bytes as f64 / total_bytes as f64) * 100.0) as u32,
                        downloaded_bytes: done_bytes,
                        total_bytes,
                        downloaded_mb,
                        total_mb: total_bytes as f64 / 1_048_576.0,
                        speed_mbps: downloaded_mb / elapsed,
                    });
                }
            }
            file.flush().await?;
        }
        Ok(())
    }
}
