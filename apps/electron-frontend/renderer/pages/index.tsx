"use client";

import React, { useEffect, useState } from "react";

// Add type definition for our electron API
declare global {
  interface Window {
    electron?: {
      getChampSelect: () => Promise<any>;
    };
  }
}

export default function Home() {
  const [isElectron, setIsElectron] = useState(false);
  const [champSelectData, setChampSelectData] = useState<any>(null);

  // Check for Electron in a useEffect
  useEffect(() => {
    setIsElectron(!!window.electron);
  }, []);

  useEffect(() => {
    if (!isElectron) {
      console.log("Not running in Electron");
      return;
    }

    // Set up polling interval
    const interval = setInterval(async () => {
      try {
        const data = await window.electron.getChampSelect();
        console.log("Champ select data:", data);
        setChampSelectData(data);
      } catch (error) {
        console.error("Failed to get champ select data:", error);
      }
    }, 1000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, [isElectron]);

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">
        League Client Integration Test
      </h1>

      {/* Display if we're running in Electron */}
      <div className="mb-4">
        Running in Electron: {isElectron ? "Yes" : "No"}
      </div>

      {/* Display the raw champ select data */}
      <div className="whitespace-pre-wrap font-mono text-sm">
        {champSelectData
          ? JSON.stringify(champSelectData, null, 2)
          : "No champ select data available"}
      </div>
    </div>
  );
}
