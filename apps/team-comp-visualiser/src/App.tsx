import { useState, useEffect } from "react";

export default function App() {
  const [dbInfo, setDbInfo] = useState<string>("Loading database info...");

  useEffect(() => {
    const fetchDbInfo = async () => {
      try {
        const info = await window.electronAPI.database.getDbInfo();
        setDbInfo(info);
      } catch (error) {
        setDbInfo(`Error fetching database info: ${error}`);
      }
    };

    fetchDbInfo();
  }, []);

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">SQLite in Electron POC</h1>
      <div className="border p-4 rounded">
        <h2 className="text-lg font-semibold mb-2">Database Info:</h2>
        <pre className="bg-gray-100 p-2 rounded whitespace-pre-wrap">
          {dbInfo}
        </pre>
      </div>
    </div>
  );
}
