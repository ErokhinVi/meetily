# In-app Update button + auto-update from personal fork — Design

Date: 2026-07-03
Status: approved by user («да делай»)

## Problem

The user runs a personal fork build of Meetily (with the global recording
hotkey). They want an in-app "Update" button and automatic updates. Meetily
already ships a full Tauri updater flow, but its endpoint points at the
UPSTREAM repo — so updating would replace the fork's custom features with
vanilla Meetily. The updater must instead pull from the user's own fork.

## Decision

Repoint the existing (already working) updater at the fork, sign releases with
the user's own key, surface a visible "Check for Updates" button in Settings,
and publish releases from the fork via a dedicated lightweight workflow.

## What already exists (unchanged)

- `UpdateCheckProvider` (auto-check on startup + listens for
  `check-updates-from-tray`), `UpdateDialog` (download/install/relaunch modal),
  `UpdateNotification` (toast), `updateService`. Tray "Check for Updates"
  dispatches the `check-updates-from-tray` DOM event.

## Changes

1. **`tauri.conf.json`**
   - `plugins.updater.endpoints` → `https://github.com/ErokhinVi/meetily/releases/latest/download/latest.json`
   - `plugins.updater.pubkey` → public key of the user's minisign keypair
     (private half stored as the fork secret `TAURI_SIGNING_PRIVATE_KEY`).
   - `bundle.createUpdaterArtifacts` → `true` (generate `.app.tar.gz` + `.sig`).
   - `version` → `0.4.1` (also bumped in `package.json`, `Cargo.toml`,
     `Cargo.lock`). The updater only offers an update when the published
     version is higher than the installed one.

2. **`PreferenceSettings.tsx`** (Settings → General)
   - New "Software Updates" section: shows current version (`getVersion()`)
     and a "Check for Updates" button that dispatches the existing
     `check-updates-from-tray` event — reusing the shared dialog, no logic
     duplicated.

3. **`.github/workflows/fork-release-macos.yml`** (new)
   - `workflow_dispatch`. Builds macOS (Apple Silicon), signs updater artifacts
     with `TAURI_SIGNING_PRIVATE_KEY` only (no Apple code-signing), and
     publishes a NON-DRAFT GitHub Release `v<version>` with dmg + app.tar.gz +
     .sig + `latest.json` via `tauri-action`. Mirrors the proven build steps
     from `build-macos.yml` (llama-helper sidecar, ffmpeg cache).
   - The stock `release.yml` is NOT used: it creates a draft (not "latest"),
     builds Windows, and requires Apple/Supabase/RSA secrets the fork lacks.

## Data flow

- One-time: user installs the v0.4.1 dmg (this build has the fork endpoint +
  user pubkey baked in). The previously handed-over v0.4.0 build still points
  at upstream and cannot self-migrate — hence the manual reinstall.
- Thereafter: bump version → run `fork-release-macos.yml` → published release
  updates `latest.json` at the "latest" URL → installed app's auto-check (or
  the Settings button) offers the update, verifies the `.sig` against the baked
  pubkey, downloads, installs, relaunches. Custom features persist because
  updates come from the fork.

## Constraints / caveats

- Fork releases are PUBLIC (GitHub Releases on a public repo). The updater's
  `/releases/latest/download/` URL requires a published, non-prerelease release.
- Requires the `TAURI_SIGNING_PRIVATE_KEY` secret on the fork (empty password).
- Builds are unsigned by Apple; first launch needs right-click → Open.
- A draft release does NOT satisfy the "latest" endpoint — the workflow
  publishes directly (releaseDraft: false).

## Testing

- `tsc --noEmit` (frontend), CI build success (Rust compiles).
- Verify `https://github.com/ErokhinVi/meetily/releases/latest/download/latest.json`
  resolves and contains a `darwin-aarch64` entry with a signature.
- Manual: install v0.4.1, later publish a v0.4.2 test release, confirm the app
  offers and applies it.
