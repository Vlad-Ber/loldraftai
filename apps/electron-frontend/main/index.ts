import { app, BrowserWindow, ipcMain } from "electron";
import * as path from "path";
import * as url from "url";
import fetch from "node-fetch";
import https from "https";

let mainWindow: BrowserWindow | null = null;

// Create an HTTPS agent that ignores self-signed certificates
const httpsAgent = new https.Agent({
  rejectUnauthorized: false,
});

// Add this function to handle League client requests
async function getChampSelect() {
  try {
    // League client uses basic auth with 'riot' as username
    const credentials = Buffer.from(
      "riot:" + process.env.LEAGUE_CLIENT_PASSWORD
    ).toString("base64");

    const response = await fetch(
      "https://127.0.0.1:2999/lol-champ-select/v1/session",
      {
        headers: {
          Authorization: `Basic ${credentials}`,
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
