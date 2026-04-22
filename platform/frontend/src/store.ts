import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ThemeMode } from './theme';

interface UIState {
  themeMode: ThemeMode;
  siderCollapsed: boolean;
  setThemeMode: (mode: ThemeMode) => void;
  toggleTheme: () => void;
  setSiderCollapsed: (v: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      themeMode: 'light',
      siderCollapsed: false,
      setThemeMode: (mode) => set({ themeMode: mode }),
      toggleTheme: () => set({ themeMode: get().themeMode === 'dark' ? 'light' : 'dark' }),
      setSiderCollapsed: (v) => set({ siderCollapsed: v }),
    }),
    { name: 'msg-ui' },
  ),
);
