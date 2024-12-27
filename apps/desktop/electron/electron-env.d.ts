/// <reference types="vite-plugin-electron/electron-env" />

declare namespace NodeJS {
  interface ProcessEnv {
    /**
     * The built directory structure
     *
     * ```tree
     * ├─┬─┬ dist
     * │ │ └── index.html
     * │ │
     * │ ├─┬ dist-electron
     * │ │ ├── main.js
     * │ │ └── preload.js
     * │
     * ```
     */
    APP_ROOT: string;
    /** /dist/ or /public/ */
    VITE_PUBLIC: string;
  }
}

interface ElectronAPI {
  on(...args: Parameters<typeof import("electron").ipcRenderer.on>): void;
  off(...args: Parameters<typeof import("electron").ipcRenderer.off>): void;
  send(...args: Parameters<typeof import("electron").ipcRenderer.send>): void;
  invoke(
    ...args: Parameters<typeof import("electron").ipcRenderer.invoke>
  ): Promise<any>;
  onUpdateNotification: (
    callback: (info: { title: string; body: string }) => void
  ) => void;
  getChampSelect: () => Promise<any>;
  storage: {
    getItem: (key: string) => Promise<any>;
    setItem: (key: string, value: any) => Promise<void>;
  };
}

declare interface Window {
  electronAPI: ElectronAPI;
}
