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
ipcMain.handle("select-db-file", async () => {
  const { dialog } = require("electron");
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [{ name: "SQLite Database", extensions: ["db"] }],
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle("get-db-info", async (_event, customPath?: string) => {
  try {
    const dbPath = customPath || DB_PATH;
    const db = new Database(dbPath);
    let dbInfo = `Database path: ${dbPath}\n\nTables:\n`;

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

// Get all champions from lookup table
ipcMain.handle("get-champions", async (_event, dbPath: string) => {
  try {
    const db = new Database(dbPath);
    const champions = db
      .prepare("SELECT id, name FROM champion_lookup ORDER BY name")
      .all();
    db.close();
    return champions;
  } catch (error) {
    throw new Error(`Failed to get champions: ${error}`);
  }
});

// Get filtered team compositions
ipcMain.handle(
  "get-team-comps",
  async (
    _event,
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
  ) => {
    try {
      const db = new Database(dbPath);

      // Build the WHERE clause based on filters
      const conditions: string[] = [];

      // Helper function to add role conditions
      const addRoleConditions = (
        rolePrefix: string,
        includeIds: number[],
        excludeIds: number[],
        columnSuffix: string
      ) => {
        if (includeIds.length > 0) {
          conditions.push(
            `${rolePrefix}_${columnSuffix}_id IN (${includeIds.join(",")})`
          );
        }
        if (excludeIds.length > 0) {
          conditions.push(
            `${rolePrefix}_${columnSuffix}_id NOT IN (${excludeIds.join(",")})`
          );
        }
      };

      // Add conditions for ally roles
      ["top", "jungle", "mid", "bot", "support"].forEach((role) => {
        addRoleConditions(
          "ally",
          filters.allyInclude[role] || [],
          filters.allyExclude[role] || [],
          role
        );
        addRoleConditions(
          "enemy",
          filters.enemyInclude[role] || [],
          filters.enemyExclude[role] || [],
          role
        );
      });

      // Build the complete query
      const whereClause =
        conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";
      const orderClause = `ORDER BY ${sort.column} ${sort.direction}`;
      const limitClause = `LIMIT ${pagination.pageSize} OFFSET ${
        (pagination.page - 1) * pagination.pageSize
      }`;

      // Get total count
      const totalCount = db
        .prepare(`SELECT COUNT(*) as count FROM team_comps ${whereClause}`)
        .get() as { count: number };

      // Get paginated results
      const results = db
        .prepare(
          `SELECT * FROM team_comps ${whereClause} ${orderClause} ${limitClause}`
        )
        .all();

      db.close();

      return {
        total: totalCount.count,
        results,
      };
    } catch (error) {
      throw new Error(`Failed to get team compositions: ${error}`);
    }
  }
);

// Add this new handler to fetch precomputed role champions
ipcMain.handle("get-role-champions", async (_event, dbPath: string) => {
  try {
    const db = new Database(dbPath);
    const roleChampions: Record<
      string,
      Array<{ id: number; name: string }>
    > = {};

    const query = `
      SELECT rc.role, c.id, c.name
      FROM role_champions rc
      JOIN champion_lookup c ON rc.champion_id = c.id
      ORDER BY rc.role, c.name
    `;

    const rows = db.prepare(query).all() as Array<{
      role: string;
      id: number;
      name: string;
    }>;

    rows.forEach(({ role, id, name }) => {
      if (!roleChampions[role]) {
        roleChampions[role] = [];
      }
      roleChampions[role].push({ id, name });
    });

    db.close();
    return roleChampions;
  } catch (error) {
    throw new Error(`Failed to get role-specific champions: ${error}`);
  }
});

function createWindow() {
  win = new BrowserWindow({
    icon: path.join(process.env.VITE_PUBLIC, "logo512.png"),
    title: "LoLDraftAI - Team Comp Visualiser",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
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
