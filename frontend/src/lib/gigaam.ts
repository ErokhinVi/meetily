// Types for GigaAM integration
export interface GigaAMModelInfo {
  name: string;
  size_mb: number;
  description?: string;
  language?: string;
  status: ModelStatus;
}

export type ModelStatus =
  | 'Available'
  | 'Missing'
  | { Downloading: number }
  | { Error: string }
  | { Corrupted: { file_size: number; expected_min_size: number } };

export interface GigaAMEngineState {
  currentModel: string | null;
  availableModels: GigaAMModelInfo[];
  isLoading: boolean;
  error: string | null;
}

// User-friendly model display configuration
export interface ModelDisplayInfo {
  friendlyName: string;
  icon: string;
  tagline: string;
  recommended?: boolean;
  tier: 'fastest' | 'balanced' | 'precise';
}

export const MODEL_DISPLAY_CONFIG: Record<string, ModelDisplayInfo> = {
  'gigaam-v3-e2e-ctc-int8': {
    friendlyName: 'Lightning',
    icon: '⚡',
    tagline: 'Быстрая • реальное время, отличная точность',
    recommended: true,
    tier: 'fastest'
  },
  'gigaam-v3-e2e-rnnt-int8': {
    friendlyName: 'Precise',
    icon: '🎯',
    tagline: 'Максимальная точность • RNN-T, чуть медленнее',
    tier: 'precise'
  }
};

// Get user-friendly display name for a model
export function getModelDisplayName(modelName: string): string {
  const displayInfo = MODEL_DISPLAY_CONFIG[modelName];
  return displayInfo?.friendlyName || modelName;
}

// Get model display info (icon, tagline, etc.)
export function getModelDisplayInfo(modelName: string): ModelDisplayInfo | null {
  return MODEL_DISPLAY_CONFIG[modelName] || null;
}

export function getStatusColor(status: ModelStatus): string {
  if (status === 'Available') return 'green';
  if (status === 'Missing') return 'gray';
  if (typeof status === 'object' && 'Downloading' in status) return 'blue';
  if (typeof status === 'object' && 'Error' in status) return 'red';
  return 'gray';
}

export function formatFileSize(sizeMb: number): string {
  if (sizeMb >= 1000) {
    return `${(sizeMb / 1000).toFixed(1)}GB`;
  }
  return `${sizeMb}MB`;
}

export function getRecommendedModel(): string {
  return 'gigaam-v3-e2e-ctc-int8';
}

// Tauri command wrappers for GigaAM backend
import { invoke } from '@tauri-apps/api/core';

export class GigaAMAPI {
  static async init(): Promise<void> {
    // GigaAM has no explicit init command; fetching available models is enough.
    await invoke('gigaam_get_available_models');
  }

  static async getAvailableModels(): Promise<GigaAMModelInfo[]> {
    return await invoke('gigaam_get_available_models');
  }

  static async loadModel(modelName: string): Promise<void> {
    await invoke('gigaam_load_model', { modelName });
  }

  static async getCurrentModel(): Promise<string | null> {
    return await invoke('gigaam_get_current_model');
  }

  static async isModelLoaded(): Promise<boolean> {
    return await invoke('gigaam_is_model_loaded');
  }

  static async getModelsDirectory(): Promise<string> {
    return await invoke('gigaam_get_models_directory');
  }

  static async downloadModel(modelName: string): Promise<void> {
    await invoke('gigaam_download_model', { modelName });
  }

  static async cancelDownload(modelName: string): Promise<void> {
    await invoke('gigaam_cancel_download', { modelName });
  }

  static async deleteCorruptedModel(modelName: string): Promise<string> {
    return await invoke('gigaam_delete_corrupted_model', { modelName });
  }

  static async hasAvailableModels(): Promise<boolean> {
    return await invoke('gigaam_has_available_models');
  }
}
