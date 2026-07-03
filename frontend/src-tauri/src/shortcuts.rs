use std::str::FromStr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Runtime};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_store::StoreExt;

// Stored alongside recording preferences, but under its own key so the
// frontend's set_recording_preferences round-trip can never wipe it.
const STORE_FILE: &str = "recording_preferences.json";
const STORE_KEY: &str = "toggle_recording_shortcut";

// Guards against key-repeat / accidental double presses toggling twice.
const TRIGGER_DEBOUNCE: Duration = Duration::from_millis(700);

static CURRENT_SHORTCUT: Mutex<Option<String>> = Mutex::new(None);
static LAST_TRIGGER: Mutex<Option<Instant>> = Mutex::new(None);

/// Called from the global-shortcut plugin handler registered in lib.rs.
pub fn handle_shortcut_event<R: Runtime>(
    app: &AppHandle<R>,
    shortcut: &Shortcut,
    state: ShortcutState,
) {
    if state != ShortcutState::Pressed {
        return;
    }

    let is_ours = CURRENT_SHORTCUT
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
        .and_then(|s| Shortcut::from_str(&s).ok())
        .map(|current| current == *shortcut)
        .unwrap_or(false);
    if !is_ours {
        return;
    }

    if let Ok(mut last) = LAST_TRIGGER.lock() {
        if let Some(prev) = *last {
            if prev.elapsed() < TRIGGER_DEBOUNCE {
                log::info!("Global shortcut: ignoring press within debounce window");
                return;
            }
        }
        *last = Some(Instant::now());
    }

    log::info!("Global shortcut pressed: toggling recording");
    // Same toggle logic as the tray menu, but without stealing focus from
    // whatever app the user is currently in.
    crate::tray::toggle_recording(app, false);
}

fn load_saved_shortcut<R: Runtime>(app: &AppHandle<R>) -> Option<String> {
    let store = app.store(STORE_FILE).ok()?;
    store
        .get(STORE_KEY)
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .filter(|s| !s.trim().is_empty())
}

/// Register the persisted shortcut (if any) at app startup.
pub fn init<R: Runtime>(app: &AppHandle<R>) {
    match load_saved_shortcut(app) {
        Some(shortcut) => match register(app, &shortcut) {
            Ok(_) => log::info!("Registered global recording shortcut: {}", shortcut),
            Err(e) => log::error!("Failed to register saved recording shortcut: {}", e),
        },
        None => log::info!("No global recording shortcut configured"),
    }
}

fn register<R: Runtime>(app: &AppHandle<R>, shortcut_str: &str) -> Result<(), String> {
    let shortcut = Shortcut::from_str(shortcut_str)
        .map_err(|e| format!("Invalid shortcut '{}': {}", shortcut_str, e))?;
    app.global_shortcut()
        .register(shortcut)
        .map_err(|e| format!("Failed to register shortcut '{}': {}", shortcut_str, e))?;
    if let Ok(mut current) = CURRENT_SHORTCUT.lock() {
        *current = Some(shortcut_str.to_string());
    }
    Ok(())
}

fn unregister_current<R: Runtime>(app: &AppHandle<R>) {
    let previous = CURRENT_SHORTCUT.lock().ok().and_then(|mut c| c.take());
    if let Some(prev) = previous {
        if let Ok(shortcut) = Shortcut::from_str(&prev) {
            if let Err(e) = app.global_shortcut().unregister(shortcut) {
                log::warn!("Failed to unregister previous shortcut '{}': {}", prev, e);
            }
        }
    }
}

#[tauri::command]
pub async fn get_toggle_recording_shortcut<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Option<String>, String> {
    Ok(load_saved_shortcut(&app))
}

#[tauri::command]
pub async fn set_toggle_recording_shortcut<R: Runtime>(
    app: AppHandle<R>,
    shortcut: Option<String>,
) -> Result<(), String> {
    let new_shortcut = shortcut.filter(|s| !s.trim().is_empty());

    // Validate before touching the currently registered shortcut, so bad
    // input doesn't leave the user without their working hotkey.
    if let Some(s) = &new_shortcut {
        Shortcut::from_str(s).map_err(|e| format!("Invalid shortcut '{}': {}", s, e))?;
    }

    unregister_current(&app);

    if let Some(s) = &new_shortcut {
        register(&app, s)?;
    }

    let store = app
        .store(STORE_FILE)
        .map_err(|e| format!("Failed to access store: {}", e))?;
    match &new_shortcut {
        Some(s) => store.set(STORE_KEY, serde_json::Value::String(s.clone())),
        None => {
            store.delete(STORE_KEY);
        }
    }
    store
        .save()
        .map_err(|e| format!("Failed to persist shortcut: {}", e))?;

    match new_shortcut {
        Some(s) => log::info!("Global recording shortcut set to: {}", s),
        None => log::info!("Global recording shortcut cleared"),
    }
    Ok(())
}
