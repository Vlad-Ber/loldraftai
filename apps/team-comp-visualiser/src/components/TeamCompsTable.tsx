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

type TeamCompsTableProps = {
  teamComps: TeamComp[];
  championMap: Record<number, string>;
};

export default function TeamCompsTable({
  teamComps,
  championMap,
}: TeamCompsTableProps) {
  // Format winrate as percentage
  const formatWinrate = (wr: number) => `${(wr * 100).toFixed(1)}%`;

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-slate-800">
            <th className="p-2 border border-slate-600">Ally Top</th>
            <th className="p-2 border border-slate-600">Ally Jungle</th>
            <th className="p-2 border border-slate-600">Ally Mid</th>
            <th className="p-2 border border-slate-600">Ally Bot</th>
            <th className="p-2 border border-slate-600">Ally Support</th>
            <th className="p-2 border border-slate-600">Enemy Top</th>
            <th className="p-2 border border-slate-600">Enemy Jungle</th>
            <th className="p-2 border border-slate-600">Enemy Mid</th>
            <th className="p-2 border border-slate-600">Enemy Bot</th>
            <th className="p-2 border border-slate-600">Enemy Support</th>
            <th className="p-2 border border-slate-600">Avg WR</th>
            <th className="p-2 border border-slate-600">Blue WR</th>
            <th className="p-2 border border-slate-600">Red WR</th>
          </tr>
        </thead>
        <tbody>
          {teamComps.map((comp) => (
            <tr key={comp.id} className="hover:bg-slate-800">
              <td className="p-2 border border-slate-600">
                {championMap[comp.ally_top_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.ally_jungle_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.ally_mid_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.ally_bot_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.ally_support_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.enemy_top_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.enemy_jungle_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.enemy_mid_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.enemy_bot_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {championMap[comp.enemy_support_id] || "Unknown"}
              </td>
              <td className="p-2 border border-slate-600">
                {formatWinrate(comp.avg_winrate)}
              </td>
              <td className="p-2 border border-slate-600">
                {formatWinrate(comp.blue_winrate)}
              </td>
              <td className="p-2 border border-slate-600">
                {formatWinrate(comp.red_winrate)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
