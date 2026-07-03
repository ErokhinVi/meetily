use crate::gigaam_engine::{DownloadProgress, GigaamEngine, ModelInfo, ModelStatus};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{command, AppHandle, Emitter, Manager, Runtime};

pub static GIGAAM_ENGINE: Mutex<Option<Arc<GigaamEngine>>> = Mutex::new(None);
static MODELS_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the shared models directory (app_data/models). Call during app setup.
pub fn set_models_directory<R: Runtime>(app: &AppHandle<R>) {
    let Ok(app_data_dir) = app.path().app_data_dir() else {
        log::error!("GigaAM: failed to get app data dir");
        return;
    };
    let models_dir = app_data_dir.join("models");
    if let Err(e) = std::fs::create_dir_all(&models_dir) {
        log::error!("GigaAM: failed to create models directory: {}", e);
        return;
    }
    *MODELS_DIR.lock().unwrap() = Some(models_dir);
}

fn get_models_directory() -> Option<PathBuf> {
    MODELS_DIR.lock().unwrap().clone()
}

fn engine() -> Option<Arc<GigaamEngine>> {
    GIGAAM_ENGINE.lock().unwrap().as_ref().cloned()
}

#[command]
pub async fn gigaam_init() -> Result<(), String> {
    let mut guard = GIGAAM_ENGINE.lock().unwrap();
    if guard.is_some() {
        return Ok(());
    }
    let engine = GigaamEngine::new_with_models_dir(get_models_directory())
        .map_err(|e| format!("Failed to initialize GigaAM engine: {}", e))?;
    *guard = Some(Arc::new(engine));
    Ok(())
}

#[command]
pub async fn gigaam_get_available_models() -> Result<Vec<ModelInfo>, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    engine
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover GigaAM models: {}", e))
}

#[command]
pub async fn gigaam_has_available_models() -> Result<bool, String> {
    let Some(engine) = engine() else {
        return Ok(false);
    };
    let models = engine
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover GigaAM models: {}", e))?;
    Ok(models.iter().any(|m| m.status == ModelStatus::Available))
}

#[command]
pub async fn gigaam_load_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    let _ = app_handle.emit(
        "gigaam-model-loading-started",
        serde_json::json!({ "modelName": model_name }),
    );
    let result = engine
        .load_model(&model_name)
        .await
        .map_err(|e| format!("Failed to load GigaAM model: {}", e));
    match &result {
        Ok(_) => {
            let _ = app_handle.emit(
                "gigaam-model-loading-completed",
                serde_json::json!({ "modelName": model_name }),
            );
        }
        Err(error) => {
            let _ = app_handle.emit(
                "gigaam-model-loading-failed",
                serde_json::json!({ "modelName": model_name, "error": error }),
            );
        }
    }
    result
}

#[command]
pub async fn gigaam_get_current_model() -> Result<Option<String>, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    Ok(engine.get_current_model().await)
}

#[command]
pub async fn gigaam_is_model_loaded() -> Result<bool, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    Ok(engine.is_model_loaded().await)
}

#[command]
pub async fn gigaam_transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    engine
        .transcribe_audio(audio_data)
        .await
        .map_err(|e| format!("GigaAM transcription failed: {}", e))
}

#[command]
pub async fn gigaam_get_models_directory() -> Result<String, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    Ok(engine.get_models_directory().await.to_string_lossy().to_string())
}

#[command]
pub async fn gigaam_download_model<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;

    let app_for_cb = app_handle.clone();
    let name_for_cb = model_name.clone();
    let callback = Box::new(move |progress: DownloadProgress| {
        let _ = app_for_cb.emit(
            "gigaam-model-download-progress",
            serde_json::json!({
                "modelName": name_for_cb,
                "progress": progress.percent,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "downloaded_mb": progress.downloaded_mb,
                "total_mb": progress.total_mb,
                "speed_mbps": progress.speed_mbps,
                "status": if progress.percent >= 100 { "completed" } else { "downloading" }
            }),
        );
    });

    engine.discover_models().await.ok();
    let result = engine
        .download_model_detailed(&model_name, Some(callback))
        .await;

    match result {
        Ok(()) => {
            let _ = app_handle.emit(
                "gigaam-model-download-complete",
                serde_json::json!({ "modelName": model_name }),
            );
            crate::tray::update_tray_menu(&app_handle);
            Ok(())
        }
        Err(e) => {
            let _ = app_handle.emit(
                "gigaam-model-download-error",
                serde_json::json!({ "modelName": model_name, "error": e.to_string() }),
            );
            Err(format!("Failed to download GigaAM model: {}", e))
        }
    }
}

#[command]
pub async fn gigaam_cancel_download<R: Runtime>(
    app_handle: AppHandle<R>,
    model_name: String,
) -> Result<(), String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    engine
        .cancel_download(&model_name)
        .await
        .map_err(|e| format!("Failed to cancel GigaAM download: {}", e))?;
    let _ = app_handle.emit(
        "gigaam-model-download-progress",
        serde_json::json!({ "modelName": model_name, "progress": 0, "status": "cancelled" }),
    );
    Ok(())
}

#[command]
pub async fn gigaam_delete_corrupted_model(model_name: String) -> Result<String, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;
    engine
        .delete_model(&model_name)
        .await
        .map_err(|e| format!("Failed to delete GigaAM model: {}", e))
}

/// Load the user's configured GigaAM model (or the first available) before recording.
pub async fn gigaam_validate_model_ready_with_config<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<String, String> {
    let engine = engine().ok_or("GigaAM engine not initialized")?;

    if engine.is_model_loaded().await {
        if let Some(current) = engine.get_current_model().await {
            return Ok(current);
        }
    }

    let configured = match crate::api::api::api_get_transcript_config(app.clone(), app.state(), None).await {
        Ok(Some(cfg)) if cfg.provider == "gigaam" && !cfg.model.is_empty() => Some(cfg.model),
        _ => None,
    };

    let models = engine
        .discover_models()
        .await
        .map_err(|e| format!("Failed to discover GigaAM models: {}", e))?;
    let available: Vec<_> = models
        .iter()
        .filter(|m| m.status == ModelStatus::Available)
        .collect();
    if available.is_empty() {
        return Err("No GigaAM models are available. Please download a model.".to_string());
    }

    let model_name = configured
        .filter(|c| available.iter().any(|m| &m.name == c))
        .unwrap_or_else(|| available[0].name.clone());

    engine
        .load_model(&model_name)
        .await
        .map_err(|e| format!("Failed to load GigaAM model {}: {}", model_name, e))?;
    Ok(model_name)
}
