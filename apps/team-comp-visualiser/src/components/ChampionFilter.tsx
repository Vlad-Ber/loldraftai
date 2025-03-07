import { useState, useEffect } from "react";

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
  return (
    <div className="p-2">
      <h4 className="text-sm font-semibold mb-1">{label}</h4>
      <select
        multiple
        value={selectedChampions.map(String)}
        onChange={(e) => {
          const selected = Array.from(e.target.selectedOptions, (option) =>
            parseInt(option.value)
          );
          onSelectionChange(selected);
        }}
        className="w-full bg-slate-800 border border-slate-600 rounded p-1 h-28 text-sm"
      >
        {champions.map((champ) => (
          <option key={champ.id} value={champ.id}>
            {champ.name}
          </option>
        ))}
      </select>
    </div>
  );
}
