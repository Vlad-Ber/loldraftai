import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/index.css";
import "./fonts.css";
import { ThemeProvider } from "./providers/theme-provider";
import { setStorageImpl } from "@draftking/ui/hooks/usePersistedState";
import App from "./App";

// Initialize storage implementation
setStorageImpl(window.electronAPI.storage);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>
);

// Use contextBridge
window.electronAPI.on("main-process-message", (_event, message) => {
  console.log(message);
});
