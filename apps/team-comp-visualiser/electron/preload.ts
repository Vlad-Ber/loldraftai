import { ipcRenderer, contextBridge } from "electron";

// --------- Expose some API to the Renderer process ---------
contextBridge.exposeInMainWorld("electronAPI", {
  on(...args: Parameters<typeof ipcRenderer.on>) {
    const [channel, listener] = args;
    return ipcRenderer.on(channel, (event, ...args) =>
      listener(event, ...args)
    );
  },
  off(...args: Parameters<typeof ipcRenderer.off>) {
    const [channel, ...omit] = args;
    return ipcRenderer.off(channel, ...omit);
  },
  send(...args: Parameters<typeof ipcRenderer.send>) {
    const [channel, ...omit] = args;
    return ipcRenderer.send(channel, ...omit);
  },
  invoke(...args: Parameters<typeof ipcRenderer.invoke>) {
    const [channel, ...omit] = args;
    return ipcRenderer.invoke(channel, ...omit);
  },

  // Simplified update-related methods
  onUpdateNotification: (
    callback: (info: { title: string; body: string }) => void
  ) => {
    ipcRenderer.on("update-notification", (_event, info) => callback(info));
  },
  getChampSelect: () => ipcRenderer.invoke("get-champ-select"),

  // Add storage methods
  storage: {
    getItem: (key: string) => ipcRenderer.invoke("electron-store-get", key),
    setItem: (key: string, value: any) =>
      ipcRenderer.invoke("electron-store-set", key, value),
  },

  database: {
    getDbInfo: (path?: string) => ipcRenderer.invoke("get-db-info", path),
    selectDbFile: () => ipcRenderer.invoke("select-db-file"),
    getChampions: (dbPath: string) =>
      ipcRenderer.invoke("get-champions", dbPath),
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
    ) =>
      ipcRenderer.invoke("get-team-comps", dbPath, filters, sort, pagination),
    getRoleChampions: (dbPath: string) =>
      ipcRenderer.invoke("get-role-champions", dbPath),
  },
});
