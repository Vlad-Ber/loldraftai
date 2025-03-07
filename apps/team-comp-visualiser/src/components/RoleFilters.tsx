import ChampionFilter from "./ChampionFilter";

type RoleFiltersProps = {
  title: string;
  champions: { id: number; name: string }[];
  roles: string[];
  includeFilters: Record<string, number[]>;
  excludeFilters: Record<string, number[]>;
  onIncludeChange: (role: string, champIds: number[]) => void;
  onExcludeChange: (role: string, champIds: number[]) => void;
  onClearFilters: (role: string) => void;
};

export default function RoleFilters({
  title,
  champions,
  roles,
  includeFilters,
  excludeFilters,
  onIncludeChange,
  onExcludeChange,
  onClearFilters,
}: RoleFiltersProps) {
  return (
    <div className="mb-4">
      <h3 className="text-lg font-bold mb-2">{title}</h3>
      <div className="flex flex-wrap">
        {roles.map((role) => (
          <div key={role} className="w-1/5 p-2">
            <div className="border border-slate-600 rounded p-2 bg-slate-900">
              <h4 className="text-base font-semibold mb-2 text-center">
                {role}
              </h4>

              <ChampionFilter
                label="Include Champions:"
                champions={champions}
                selectedChampions={includeFilters[role] || []}
                onSelectionChange={(ids) => onIncludeChange(role, ids)}
              />

              <ChampionFilter
                label="Exclude Champions:"
                champions={champions}
                selectedChampions={excludeFilters[role] || []}
                onSelectionChange={(ids) => onExcludeChange(role, ids)}
              />

              <button
                onClick={() => onClearFilters(role)}
                className="w-full mt-2 py-1 bg-slate-700 text-white rounded hover:bg-slate-600 text-sm"
              >
                Clear Filters
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
