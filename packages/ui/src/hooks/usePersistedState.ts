import { useState, useEffect } from 'react';
import type { Elo } from '../lib/types';

interface Storage {
  getItem(key: string): string | null | Promise<string | null>;
  setItem(key: string, value: string): void | Promise<void>;
}

// Default to localStorage, can be overridden for electron
let storage: Storage = typeof window !== 'undefined' ? window.localStorage : {
  getItem: () => null,
  setItem: () => {},
};

export const setStorageImpl = (impl: Storage) => {
  storage = impl;
};

export function usePersistedState<T>(key: string, defaultValue: T) {
  const [value, setValue] = useState<T>(defaultValue);
  const [isInitialized, setIsInitialized] = useState(false);

  // Initial load
  useEffect(() => {
    const loadInitialValue = async () => {
      try {
        const stored = await Promise.resolve(storage.getItem(key));
        if (stored) {
          setValue(JSON.parse(stored));
        }
      } catch (error) {
        console.error(`Error loading persisted value for ${key}:`, error);
      } finally {
        setIsInitialized(true);
      }
    };
    
    void loadInitialValue();
  }, [key]);

  // Save on changes
  useEffect(() => {
    if (!isInitialized) return;

    const saveValue = async () => {
      try {
        await Promise.resolve(storage.setItem(key, JSON.stringify(value)));
      } catch (error) {
        console.error(`Error saving persisted value for ${key}:`, error);
      }
    };

    void saveValue();
  }, [key, value, isInitialized]);

  return [value, setValue] as const;
}

export const usePersistedElo = (defaultElo: Elo = 'emerald') => {
  return usePersistedState<Elo>('draftking-elo', defaultElo);
};

export const usePersistedPatch = (defaultPatch: string) => {
  return usePersistedState<string>('draftking-patch', defaultPatch);
}; 