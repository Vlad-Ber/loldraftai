import { useState, useEffect } from "react";

export default function App() {
  const [dbInfo, setDbInfo] = useState<string>("No database loaded");
  const [dbPath, setDbPath] = useState<string | null>(null);

  const handleSelectFile = async () => {
    try {
      const selectedPath = await window.electronAPI.database.selectDbFile();
      if (selectedPath) {
        setDbPath(selectedPath);
        const info = await window.electronAPI.database.getDbInfo(selectedPath);
        setDbInfo(info);
      }
    } catch (error) {
      setDbInfo(`Error selecting database: ${error}`);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">SQLite Database Viewer</h1>

      <div className="mb-4">
        <button
          onClick={handleSelectFile}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Select Database File
        </button>
        {dbPath && (
          <p className="mt-2 text-sm text-gray-600">
            Current database: {dbPath}
          </p>
        )}
      </div>

      <div className="border p-4 rounded">
        <h2 className="text-lg font-semibold mb-2">Database Info:</h2>
        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">
          {dbInfo}
        </pre>
      </div>
    </div>
  );
}
