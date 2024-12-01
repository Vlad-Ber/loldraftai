import { app, BrowserWindow, ipcMain } from "electron";
import { exec } from "child_process";
import * as path from "path";
import fetch from "node-fetch";
import https from "https";
import * as fs from "fs";
import * as os from "os";
import * as url from "url";

let mainWindow: BrowserWindow | null = null;

interface LeagueCredentials {
  port: string;
  password: string;
}

async function getLeagueCredentials(): Promise<LeagueCredentials | null> {
  const platform = os.platform();

  if (platform === "darwin") {
    // macOS: Read lockfile from default installation path
    try {
      const lockfilePath = path.join(
        "/Applications",
        "League of Legends.app",
        "Contents",
        "LoL",
        "lockfile"
      );
      const lockfile = fs.readFileSync(lockfilePath, "utf8");
      const [, , port, password] = lockfile.split(":");
      return { port, password };
    } catch (error) {
      console.error("Error reading lockfile:", error);
      return null;
    }
  } else if (platform === "win32") {
    // Windows: Use wmic to get credentials from command line
    return new Promise((resolve) => {
      exec(
        "wmic PROCESS WHERE name='LeagueClientUx.exe' GET commandline",
        (error, stdout) => {
          if (error) {
            console.error("Error getting League process:", error);
            resolve(null);
            return;
          }

          const port = stdout.match(/--app-port=([0-9]+)/)?.[1];
          const password = stdout.match(/--remoting-auth-token=([\w-]+)/)?.[1];

          if (port && password) {
            resolve({ port, password });
          } else {
            resolve(null);
          }
        }
      );
    });
  }

  console.error("Unsupported platform:", platform);
  return null;
}

// Create an HTTPS agent that ignores self-signed certificates
const httpsAgent = new https.Agent({
  rejectUnauthorized: false,
});

async function getChampSelect() {
  try {
    const credentials = await getLeagueCredentials();
    if (!credentials) {
      throw new Error("League client not running or credentials not found");
    }

    const { port, password } = credentials;
    const auth = Buffer.from(`riot:${password}`).toString("base64");

    const response = await fetch(
      `https://127.0.0.1:${port}/lol-champ-select/v1/session`,
      {
        headers: {
          Authorization: `Basic ${auth}`,
        },
        agent: httpsAgent,
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Error fetching champ select:", error);
    return null;
  }
}

// Register the IPC handler
ipcMain.handle("get-champ-select", getChampSelect);

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });

  // Set NODE_ENV for development
  process.env.NODE_ENV = process.env.NODE_ENV || "development";

  // In development, use the hosted version of your Next.js app
  if (process.env.NODE_ENV === "development") {
    mainWindow.loadURL("http://localhost:3000");
    mainWindow.webContents.openDevTools();
  } else {
    // In production, use the built Next.js app
    mainWindow.loadURL(
      url.format({
        pathname: path.join(__dirname, "../renderer/out/index.html"),
        protocol: "file:",
        slashes: true,
      })
    );
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});
