import { contextBridge, ipcRenderer } from "electron";

// Expose specific Electron APIs to the renderer process
contextBridge.exposeInMainWorld("electron", {
  getChampSelect: () => ipcRenderer.invoke("get-champ-select"),
});
