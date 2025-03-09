type ChampionFilterProps = {
  label: string;
  champions: { id: number; name: string }[];
  selectedChampions: number[];
  onSelectionChange: (champIds: number[]) => void;
};

export default function ChampionFilter({
  label,
  champions,
  selectedChampions,
  onSelectionChange,
}: ChampionFilterProps) {
  const toggleChampion = (champId: number) => {
    if (selectedChampions.includes(champId)) {
      onSelectionChange(selectedChampions.filter((id) => id !== champId));
    } else {
      onSelectionChange([...selectedChampions, champId]);
    }
  };

  return (
    <div className="p-2">
      <h4 className="text-sm font-semibold mb-1">{label}</h4>
      <div className="w-full bg-slate-800 border border-slate-600 rounded p-1 h-28 text-sm overflow-y-auto">
        {champions.map((champ) => (
          <div
            key={champ.id}
            onClick={() => toggleChampion(champ.id)}
            className={`px-1 py-0.5 cursor-pointer hover:bg-gray-500 ${
              selectedChampions.includes(champ.id)
                ? "bg-gray-500 font-medium"
                : ""
            }`}
          >
            {champ.name}
          </div>
        ))}
      </div>
    </div>
  );
}
