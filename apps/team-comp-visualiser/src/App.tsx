import { useState, useEffect } from "react";
import RoleFilters from "./components/RoleFilters";
import TeamCompsTable from "./components/TeamCompsTable";
import Pagination from "./components/Pagination";

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

// Define role mappings
const ALLY_ROLES = ["top", "jungle", "mid", "bot", "support"];
const ENEMY_ROLES = ["top", "jungle", "mid", "bot", "support"];
const ROLE_DISPLAY_NAMES: Record<string, string> = {
  top: "Top",
  jungle: "Jungle",
  mid: "Mid",
  bot: "Bot",
  support: "Support",
};

// Add this type definition
type RoleChampions = Record<string, Array<{ id: number; name: string }>>;

export default function App() {
  // State for database
  const [dbPath, setDbPath] = useState<string | null>(null);

  // State for champions
  const [champions, setChampions] = useState<Champion[]>([]);
  const [championMap, setChampionMap] = useState<Record<number, string>>({});

  // State for filters
  const [allyIncludeFilters, setAllyIncludeFilters] = useState<
    Record<string, number[]>
  >({});
  const [allyExcludeFilters, setAllyExcludeFilters] = useState<
    Record<string, number[]>
  >({});
  const [enemyIncludeFilters, setEnemyIncludeFilters] = useState<
    Record<string, number[]>
  >({});
  const [enemyExcludeFilters, setEnemyExcludeFilters] = useState<
    Record<string, number[]>
  >({});

  // State for sorting
  const [sortColumn, setSortColumn] = useState<
    "avg_winrate" | "blue_winrate" | "red_winrate"
  >("avg_winrate");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");

  // State for pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalItems, setTotalItems] = useState(0);

  // State for team comps
  const [teamComps, setTeamComps] = useState<TeamComp[]>([]);

  // State for loading
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add state for role-specific champions
  const [roleChampions, setRoleChampions] = useState<RoleChampions>({});

  // Update handleSelectFile to load role champions
  const handleSelectFile = async () => {
    try {
      const selectedPath = await window.electronAPI.database.selectDbFile();
      if (selectedPath) {
        setDbPath(selectedPath);

        // Load champions for champion map
        await loadChampions(selectedPath);

        // Load precomputed role-specific champions
        await loadRoleChampions(selectedPath);

        // Fetch initial team comps
        await fetchTeamComps(selectedPath);
      }
    } catch (error) {
      setError(`Error selecting database: ${error}`);
    }
  };

  // Load champions from database
  const loadChampions = async (path: string) => {
    try {
      const champs = await window.electronAPI.database.getChampions(path);
      setChampions(champs);

      // Create champion map for quick lookup
      const champMap: Record<number, string> = {};
      champs.forEach((champ) => {
        champMap[champ.id] = champ.name;
      });
      setChampionMap(champMap);

      return champs;
    } catch (error) {
      setError(`Error loading champions: ${error}`);
      return [];
    }
  };

  // Add function to load role-specific champions
  const loadRoleChampions = async (path: string) => {
    try {
      const champions = await window.electronAPI.database.getRoleChampions(
        path
      );
      setRoleChampions(champions);
    } catch (error) {
      setError(`Error loading role-specific champions: ${error}`);
    }
  };

  // Fetch team comps with current filters
  const fetchTeamComps = async (path: string = dbPath!) => {
    if (!path) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await window.electronAPI.database.getTeamComps(
        path,
        {
          allyInclude: allyIncludeFilters,
          allyExclude: allyExcludeFilters,
          enemyInclude: enemyIncludeFilters,
          enemyExclude: enemyExcludeFilters,
        },
        {
          column: sortColumn,
          direction: sortDirection,
        },
        {
          page: currentPage,
          pageSize: pageSize,
        }
      );

      setTeamComps(result.results);
      setTotalItems(result.total);
    } catch (error) {
      setError(`Error fetching team compositions: ${error}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Calculate total pages
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  // Handle changes to inclusion/exclusion filters
  const handleAllyIncludeChange = (role: string, champIds: number[]) => {
    setAllyIncludeFilters((prev) => ({ ...prev, [role]: champIds }));
    setCurrentPage(1);
  };

  const handleAllyExcludeChange = (role: string, champIds: number[]) => {
    setAllyExcludeFilters((prev) => ({ ...prev, [role]: champIds }));
    setCurrentPage(1);
  };

  const handleEnemyIncludeChange = (role: string, champIds: number[]) => {
    setEnemyIncludeFilters((prev) => ({ ...prev, [role]: champIds }));
    setCurrentPage(1);
  };

  const handleEnemyExcludeChange = (role: string, champIds: number[]) => {
    setEnemyExcludeFilters((prev) => ({ ...prev, [role]: champIds }));
    setCurrentPage(1);
  };

  // Handle clearing filters for a role
  const handleClearAllyFilters = (role: string) => {
    setAllyIncludeFilters((prev) => ({ ...prev, [role]: [] }));
    setAllyExcludeFilters((prev) => ({ ...prev, [role]: [] }));
    setCurrentPage(1);
  };

  const handleClearEnemyFilters = (role: string) => {
    setEnemyIncludeFilters((prev) => ({ ...prev, [role]: [] }));
    setEnemyExcludeFilters((prev) => ({ ...prev, [role]: [] }));
    setCurrentPage(1);
  };

  // Handle page changes
  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  // Handle page size changes
  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    setCurrentPage(1);
  };

  // Handle sort changes
  const handleSortChange = (
    column: "avg_winrate" | "blue_winrate" | "red_winrate"
  ) => {
    if (sortColumn === column) {
      // Toggle direction if same column
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      // New column, default to descending
      setSortColumn(column);
      setSortDirection("desc");
    }
    setCurrentPage(1);
  };

  // Effect to fetch team comps when filters, sort, or pagination changes
  useEffect(() => {
    if (dbPath) {
      fetchTeamComps();
    }
  }, [
    dbPath,
    allyIncludeFilters,
    allyExcludeFilters,
    enemyIncludeFilters,
    enemyExcludeFilters,
    sortColumn,
    sortDirection,
    currentPage,
    pageSize,
  ]);

  return (
    <div className="p-4 text-white bg-slate-950 min-h-screen">
      <h1 className="text-2xl font-bold mb-4">
        LoL Team Composition Visualizer
      </h1>

      {!dbPath ? (
        <div className="text-center py-12">
          <p className="mb-4">Select a database file to get started</p>
          <button
            onClick={handleSelectFile}
            className="px-6 py-3 bg-blue-700 text-white rounded hover:bg-blue-600"
          >
            Select Database File
          </button>
        </div>
      ) : (
        <>
          <div className="mb-4 flex justify-between items-center">
            <p className="text-sm">
              Current database: <span className="font-mono">{dbPath}</span>
            </p>
            <button
              onClick={handleSelectFile}
              className="px-3 py-1 bg-blue-700 text-white rounded hover:bg-blue-600"
            >
              Change Database
            </button>
          </div>

          {error && (
            <div className="p-3 mb-4 bg-red-900 border border-red-700 rounded">
              {error}
            </div>
          )}

          <div className="mb-6">
            <div className="mb-4">
              <h2 className="text-xl font-bold mb-2">Filters</h2>

              <RoleFilters
                title="Ally Team"
                champions={champions}
                roleChampions={roleChampions}
                teamPrefix="ally"
                roles={ALLY_ROLES}
                includeFilters={allyIncludeFilters}
                excludeFilters={allyExcludeFilters}
                onIncludeChange={handleAllyIncludeChange}
                onExcludeChange={handleAllyExcludeChange}
                onClearFilters={handleClearAllyFilters}
              />

              <RoleFilters
                title="Enemy Team"
                champions={champions}
                roleChampions={roleChampions}
                teamPrefix="enemy"
                roles={ENEMY_ROLES}
                includeFilters={enemyIncludeFilters}
                excludeFilters={enemyExcludeFilters}
                onIncludeChange={handleEnemyIncludeChange}
                onExcludeChange={handleEnemyExcludeChange}
                onClearFilters={handleClearEnemyFilters}
              />
            </div>

            <div className="mb-4">
              <h3 className="text-lg font-bold mb-2">Sort By</h3>
              <div className="flex gap-4">
                <button
                  onClick={() => handleSortChange("avg_winrate")}
                  className={`px-3 py-1 rounded ${
                    sortColumn === "avg_winrate"
                      ? "bg-blue-700"
                      : "bg-slate-700 hover:bg-slate-600"
                  }`}
                >
                  Average Winrate{" "}
                  {sortColumn === "avg_winrate" &&
                    (sortDirection === "desc" ? "↓" : "↑")}
                </button>
                <button
                  onClick={() => handleSortChange("blue_winrate")}
                  className={`px-3 py-1 rounded ${
                    sortColumn === "blue_winrate"
                      ? "bg-blue-700"
                      : "bg-slate-700 hover:bg-slate-600"
                  }`}
                >
                  Blue Side Winrate{" "}
                  {sortColumn === "blue_winrate" &&
                    (sortDirection === "desc" ? "↓" : "↑")}
                </button>
                <button
                  onClick={() => handleSortChange("red_winrate")}
                  className={`px-3 py-1 rounded ${
                    sortColumn === "red_winrate"
                      ? "bg-blue-700"
                      : "bg-slate-700 hover:bg-slate-600"
                  }`}
                >
                  Red Side Winrate{" "}
                  {sortColumn === "red_winrate" &&
                    (sortDirection === "desc" ? "↓" : "↑")}
                </button>
              </div>
            </div>
          </div>

          {isLoading ? (
            <div className="text-center py-8">Loading...</div>
          ) : teamComps.length > 0 ? (
            <>
              <TeamCompsTable teamComps={teamComps} championMap={championMap} />
              <Pagination
                currentPage={currentPage}
                totalPages={totalPages}
                pageSize={pageSize}
                totalItems={totalItems}
                onPageChange={handlePageChange}
                onPageSizeChange={handlePageSizeChange}
              />
            </>
          ) : (
            <div className="text-center py-8">
              No team compositions match your filters
            </div>
          )}
        </>
      )}
    </div>
  );
}
