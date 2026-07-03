// audio/transcription/gigaam_provider.rs
//
// GigaAM-v3 (Sber) transcription provider. Russian-only.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use std::sync::Arc;

pub struct GigaamProvider {
    engine: Arc<crate::gigaam_engine::GigaamEngine>,
}

impl GigaamProvider {
    pub fn new(engine: Arc<crate::gigaam_engine::GigaamEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl TranscriptionProvider for GigaamProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        _language: Option<String>, // GigaAM-v3 is Russian-only; language hint ignored
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        match self.engine.transcribe_audio(audio).await {
            Ok(text) => Ok(TranscriptResult {
                text: text.trim().to_string(),
                confidence: None,
                is_partial: false,
            }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded().await
    }

    async fn get_current_model(&self) -> Option<String> {
        self.engine.get_current_model().await
    }

    fn provider_name(&self) -> &'static str {
        "GigaAM"
    }
}
