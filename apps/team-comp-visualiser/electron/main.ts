import { app, BrowserWindow, ipcMain, Menu } from "electron";
import { fileURLToPath } from "node:url";
import path from "node:path";
import Store from "electron-store";
import Database from "better-sqlite3";
import fs from "fs";

// const require = createRequire(import.meta.url)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// The built directory structure
//
// ├─┬─┬ dist
// │ │ └── index.html
// │ │
// │ ├─┬ dist-electron
// │ │ ├── main.js
// │ │ └── preload.mjs
// │
process.env.APP_ROOT = path.join(__dirname, "..");

// 🚧 Use ['ENV_NAME'] avoid vite:define plugin - Vite@2.x
export const VITE_DEV_SERVER_URL = process.env["VITE_DEV_SERVER_URL"];
export const MAIN_DIST = path.join(process.env.APP_ROOT, "dist-electron");
export const RENDERER_DIST = path.join(process.env.APP_ROOT, "dist");

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL
  ? path.join(process.env.APP_ROOT, "public")
  : RENDERER_DIST;

let win: BrowserWindow | null;

const store = new Store();

// Add IPC handlers for electron-store
ipcMain.handle("electron-store-get", (_event, key) => {
  return store.get(key);
});

ipcMain.handle("electron-store-set", (_event, key, value) => {
  store.set(key, value);
});

// Define database path
const DB_PATH = path.join(app.getPath("userData"), "test.db");

// Add SQLite-related IPC handlers
ipcMain.handle("get-db-info", async () => {
  try {
    const db = new Database(DB_PATH);
    let dbInfo = `Database path: ${DB_PATH}\n\nTables:\n`;

    // Get list of tables
    const tables = db
      .prepare("SELECT name FROM sqlite_master WHERE type='table'")
      .all() as Array<{ name: string }>;
    dbInfo += tables.map((t: { name: string }) => t.name).join("\n");

    // Get first row from test_table if it exists
    if (tables.some((t: { name: string }) => t.name === "test_table")) {
      const row = db.prepare("SELECT * FROM test_table LIMIT 1").get();
      dbInfo += `\n\nFirst row from test_table:\n${JSON.stringify(
        row,
        null,
        2
      )}`;
    }

    db.close();
    return dbInfo;
  } catch (error) {
    return `Error accessing database: ${
      error instanceof Error ? error.message : "Unknown error"
    }`;
  }
});

function createWindow() {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "logo512.png"),
    title: "LoLDraftAI - Team Comp Visualiser",
    webPreferences: {
      preload: path.join(__dirname, "preload.mjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: "#09090b",
    show: false,
  });

  // Show window when ready to render
  win.once("ready-to-show", () => {
    win?.show();
    win?.maximize();
  });

  // Remove default menu
  Menu.setApplicationMenu(null);

  win.maximize();

  // Test active push message to Renderer-process.
  win.webContents.on("did-finish-load", () => {
    win?.webContents.send("main-process-message", new Date().toLocaleString());
  });

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL);
  } else {
    // win.loadFile('dist/index.html')
    win.loadFile(path.join(RENDERER_DIST, "index.html"));
  }
}

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
    win = null;
  }
});

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

app.whenReady().then(() => {
  createWindow();

  // Create a test database if it doesn't exist
  if (!fs.existsSync(DB_PATH)) {
    try {
      const db = new Database(DB_PATH);
      db.exec(
        "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
      );

      const insert = db.prepare("INSERT INTO test_table (name) VALUES (?)");
      insert.run("Test Entry 1");
      insert.run("Test Entry 2");

      db.close();
      console.log("Created test database at:", DB_PATH);
    } catch (error) {
      console.error("Error creating database:", error);
    }
  }
});
