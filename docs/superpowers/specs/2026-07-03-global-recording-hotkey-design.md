# Global Recording Hotkey — Design

Date: 2026-07-03
Status: approved implicitly by user request («хочу hotkey включения/выключения записи, чтобы я мог его назначить — доделай»)

## Problem

Meetily has no keyboard shortcut to start/stop recording. The only quick control
is the tray menu. Users in a meeting (another app focused) want a single global
key combo to toggle recording without switching to Meetily.

## Decision

Global (system-wide), user-assignable shortcut, handled on the Rust side via the
official `tauri-plugin-global-shortcut`, reusing the existing tray toggle logic.

Alternatives considered:
- **Frontend-registered shortcut (JS plugin API)** — would duplicate the tray's
  start/stop orchestration in TS and need extra capability permissions. Rejected.
- **In-app (webview) key listener** — not global; useless when another app is
  focused, which is the primary use case. Rejected.

## Components

1. **`src-tauri/src/shortcuts.rs`** (new)
   - Persists the accelerator string in the `recording_preferences.json` store
     under its own key `toggle_recording_shortcut` (not inside the
     `RecordingPreferences` struct, so the frontend's preferences round-trip
     can't wipe it).
   - `init()` registers the saved shortcut at startup.
   - Commands `get_toggle_recording_shortcut` / `set_toggle_recording_shortcut`
     (validate → unregister old → register new → persist). Registration errors
     (e.g. combo taken by another app) surface to the UI as toasts.
   - `handle_shortcut_event()` — called from the plugin's global handler; checks
     the shortcut matches, debounces (700 ms), then calls
     `tray::toggle_recording(app, /*focus_window=*/false)` so the hotkey never
     steals focus from the current app.

2. **`src-tauri/src/tray.rs`** — `toggle_recording_handler` refactored into
   `pub(crate) toggle_recording(app, focus_window)`; tray passes `true`.

3. **`src/components/RecordingSettings.tsx`** — "Global Recording Shortcut"
   row: capture-mode button (records the next key combo; Esc cancels; requires
   ≥1 modifier or an F-key), Clear button, macOS-symbol display (⌘⇧R).
   Accelerator tokens derived from `KeyboardEvent.code` (`KeyR`→`R`,
   `Digit5`→`5`, otherwise the W3C code name, all parseable by `global_hotkey`).

## Defaults & error handling

- No default combo: the shortcut is off until the user assigns one (avoids
  clobbering other apps' hotkeys).
- Invalid/duplicate registration → command returns Err(String) → toast; the
  previous shortcut is only dropped after the new one validates.
- Rapid double-press → 700 ms debounce in Rust.

## Testing

- `cargo check` for the Rust side; `tsc --noEmit` for the frontend.
- Manual: assign combo in Settings → Recording, press it while another app is
  focused, verify recording starts (tray state changes) and stops on second
  press; restart app and verify the shortcut still works.
