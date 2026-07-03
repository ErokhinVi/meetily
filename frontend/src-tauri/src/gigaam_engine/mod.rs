//! GigaAM-v3 (Sber) Russian ASR engine — ONNX e2e-ctc via `ort`.

pub mod commands;
pub mod gigaam_engine;
pub mod model;

pub use gigaam_engine::{DownloadProgress, GigaamEngine, ModelInfo, ModelStatus};
pub use model::GigaamKind;
