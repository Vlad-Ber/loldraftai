import { useState, useEffect } from "react";

type Champion = {
  id: number;
  name: string;
};

type TeamComp = {
  id: number;
  ally_top_id: number;
  ally_jungle_id: number;
  ally_mid_id: number;
  ally_bot_id: number;
  ally_support_id: number;
  enemy_top_id: number;
  enemy_jungle_id: number;
  enemy_mid_id: number;
  enemy_bot_id: number;
  enemy_support_id: number;
  blue_winrate: number;
  red_winrate: number;
  avg_winrate: number;
};

export default function App() {
  const [dbInfo, setDbInfo] = useState<string>("No database loaded");
  const [dbPath, setDbPath] = useState<string | null>(null);
  const [champions, setChampions] = useState<Champion[]>([]);
  const [teamComps, setTeamComps] = useState<TeamComp[]>([]);

  const handleSelectFile = async () => {
    try {
      const selectedPath = await window.electronAPI.database.selectDbFile();
      if (selectedPath) {
        setDbPath(selectedPath);

        // Get basic DB info
        const info = await window.electronAPI.database.getDbInfo(selectedPath);
        setDbInfo(info);

        // Try to get champions
        try {
          const champs = await window.electronAPI.database.getChampions(
            selectedPath
          );
          setChampions(champs);
          console.log("Loaded champions:", champs.length);
        } catch (error) {
          console.error("Error loading champions:", error);
        }

        // Try a sample team comp query
        try {
          const comps = await window.electronAPI.database.getTeamComps(
            selectedPath,
            {
              allyInclude: {},
              allyExclude: {},
              enemyInclude: {},
              enemyExclude: {},
            },
            {
              column: "avg_winrate",
              direction: "desc",
            },
            {
              page: 1,
              pageSize: 5,
            }
          );
          setTeamComps(comps.results);
          console.log("Sample query results:", comps);
        } catch (error) {
          console.error("Error loading team comps:", error);
        }
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

      <div className="border p-4 rounded mb-4">
        <h2 className="text-lg font-semibold mb-2">Database Info:</h2>
        <pre className=" p-2 rounded whitespace-pre-wrap">{dbInfo}</pre>
      </div>

      {champions.length > 0 && (
        <div className="border p-4 rounded mb-4">
          <h2 className="text-lg font-semibold mb-2">Champions Loaded:</h2>
          <pre className=" p-2 rounded whitespace-pre-wrap">
            {JSON.stringify(champions.slice(0, 5), null, 2)}{" "}
            {/* Show first 5 champions */}
            {champions.length > 5 && `\n... and ${champions.length - 5} more`}
          </pre>
        </div>
      )}

      {teamComps.length > 0 && (
        <div className="border p-4 rounded">
          <h2 className="text-lg font-semibold mb-2">
            Sample Team Comps (Top 5):
          </h2>
          <pre className=" p-2 rounded whitespace-pre-wrap">
            {JSON.stringify(teamComps, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
