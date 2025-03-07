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
  database: {
    getDbInfo: (path?: string) => Promise<string>;
    selectDbFile: () => Promise<string | null>;
    getChampions: (
      dbPath: string
    ) => Promise<Array<{ id: number; name: string }>>;
    getTeamComps: (
      dbPath: string,
      filters: {
        allyInclude: { [role: string]: number[] };
        allyExclude: { [role: string]: number[] };
        enemyInclude: { [role: string]: number[] };
        enemyExclude: { [role: string]: number[] };
      },
      sort: {
        column: "avg_winrate" | "blue_winrate" | "red_winrate";
        direction: "asc" | "desc";
      },
      pagination: {
        page: number;
        pageSize: number;
      }
    ) => Promise<{
      total: number;
      results: Array<{
        id: number;
        ally_top_id: number;
        ally_jungle_id: number;
        ally_mid_id: number;
        ally_bot_id: number;
        ally_support_id: number;
        enemy_top_id: number;
        enemy_jungle_id: number;
        enemy_mid_id: number;
        enemy_bot_id: number;
        enemy_support_id: number;
        blue_winrate: number;
        red_winrate: number;
        avg_winrate: number;
      }>;
    }>;
  };
}

declare interface Window {
  electronAPI: ElectronAPI;
}
