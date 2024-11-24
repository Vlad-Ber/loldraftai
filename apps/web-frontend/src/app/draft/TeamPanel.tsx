import React, { useState } from "react";
import Image from "next/image";
import clsx from "clsx";
import { TrashIcon } from "@heroicons/react/16/solid";

import type {
  Team,
  TeamIndex,
  ChampionIndex,
  SelectedSpot,
  Champion,
} from "@/app/types";

const roles = ["Top", "Jungle", "Mid", "Bot", "Support"];

interface TeamTitleProps {
  is_blue_side: boolean;
}

const TeamTitle = ({ is_blue_side }: TeamTitleProps) => (
  <div
    className={clsx("rounded-lg p-4 shadow text-center", {
      "bg-blue-500": is_blue_side,
      "bg-red-500": !is_blue_side,
    })}
  >
    <h5 className="text-lg font-bold text-white">
      {is_blue_side ? "Blue Side" : "Red Side"}
    </h5>
  </div>
);
interface RoleAvatarProps {
  role: string;
}
const RoleAvatar = ({ role }: RoleAvatarProps) => (
  <div className="list-item-avatar">
    <Image
      src={`/icons/roles/Position_Challenger-${role}.png`}
      alt={role}
      width={80}
      height={80}
    />
  </div>
);

interface ChampionAvatarProps {
  teamMember: { name: string; icon: string };
  index: ChampionIndex;
  setHoveredChampion: (index: number | null) => void;
  hoveredChampion: number | null;
  handleDeleteChampion: (
    index: ChampionIndex,
    event: React.MouseEvent<HTMLLIElement>
  ) => void;
}

const ChampionAvatar = ({
  teamMember,
  index,
  setHoveredChampion,
  handleDeleteChampion,
}: ChampionAvatarProps) => {
  const menuRef = React.useRef<HTMLDivElement>(null);
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [menuPosition, setMenuPosition] = React.useState<{
    x: number;
    y: number;
  }>({ x: 0, y: 0 });
  const open = Boolean(anchorEl);

  const handleContextMenu = (event: React.MouseEvent<HTMLElement>) => {
    event.preventDefault();
    setAnchorEl(event.currentTarget);

    // Calculate position with screen boundary consideration including window scroll
    let x = event.clientX + window.scrollX;
    let y = event.clientY + window.scrollY;
    //TODO: hardcorded width and height is hacky
    const menuWidth = 300; // Assuming the menu width
    const menuHeight = 150; // Assuming the menu height
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;

    // Correct position if the menu goes outside of the screen
    if (x + menuWidth > screenWidth) {
      x -= x + menuWidth - screenWidth;
    }
    if (y + menuHeight > screenHeight) {
      y -= y + menuHeight - screenHeight;
    }

    const correctedMenuPosition = { x, y };
    setMenuPosition(correctedMenuPosition);
  };

  React.useEffect(() => {
    const checkIfClickedOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setAnchorEl(null);
      }
    };

    document.addEventListener("mousedown", checkIfClickedOutside);

    return () => {
      document.removeEventListener("mousedown", checkIfClickedOutside);
    };
  }, []);

  return (
    <div
      onMouseEnter={() => setHoveredChampion(index)}
      onMouseLeave={() => setHoveredChampion(null)}
      onContextMenu={handleContextMenu}
      className="flex cursor-pointer items-center justify-between"
    >
      <Image
        src={`/icons/champions/${teamMember.icon}`}
        alt={teamMember.name}
        className="block"
        width={80}
        height={80}
      />
      {open && (
        <div
          ref={menuRef}
          id="context-menu"
          className="absolute rounded bg-white text-black shadow"
          style={{
            top: `${menuPosition.y}px`,
            left: `${menuPosition.x}px`,
            zIndex: 1000,
          }}
        >
          <ul className="m-0 list-none p-0">
            <li
              className="flex cursor-pointer items-center rounded p-2 hover:bg-gray-100"
              onClick={(event) => {
                handleDeleteChampion(index, event);
              }}
            >
              <TrashIcon className="mr-2 h-4 w-4" /> Remove Champion
            </li>
          </ul>
        </div>
      )}
    </div>
  );
};

interface RoleListItemProps {
  role: string;
  teamMember: Champion | undefined;
  is_first_team: boolean;
  index: ChampionIndex;
  pannelTeamIndex: number;
  selectedSpot: SelectedSpot | null;
  handleSpotClick: (index: ChampionIndex) => void;
  setHoveredChampion: (index: number | null) => void;
  hoveredChampion: number | null;
  handleDeleteChampion: (index: ChampionIndex, event: React.MouseEvent) => void;
}

const RoleListItem = ({ role, teamMember, ...props }: RoleListItemProps) => {
  const isSelected =
    (props.selectedSpot &&
      props.selectedSpot.championIndex === props.index &&
      props.selectedSpot.teamIndex === props.pannelTeamIndex) ??
    false;
  //const boxShadowStyle = isSelected ? "shadow-inner shadow-yellow-300" : "hover:shadow-lg hover:shadow-yellow-300";
  const flexDirection = props.is_first_team ? "flex-row" : "flex-row-reverse";

  const borderStyle = isSelected
    ? "border-4 border-yellow-300"
    : "hover:border-2 hover:border-yellow-300";
  //const backgroundColor = isSelected ? "bg-yellow-100" : "hover:bg-yellow-50";
  //const scale = isSelected ? "transform scale-105" : "hover:scale-100";
  //const opacity = isSelected ? "opacity-100" : "hover:opacity-75";
  //const textStyle = isSelected ? "text-yellow-600 font-semibold" : "hover:text-yellow-400";

  return (
    <div
      className={`flex justify-center p-1 ${borderStyle}`}
      onClick={() => props.handleSpotClick(props.index)}
    >
      <div className={`flex ${flexDirection} items-center`}>
        {!teamMember && <RoleAvatar role={role} />}
        {teamMember && (
          <ChampionAvatar
            teamMember={teamMember}
            index={props.index}
            setHoveredChampion={props.setHoveredChampion}
            hoveredChampion={props.hoveredChampion}
            handleDeleteChampion={props.handleDeleteChampion}
          />
        )}
      </div>
    </div>
  );
};

interface TeamPanelProps {
  team: Team;
  is_first_team: boolean;
  onDeleteChampion: (index: ChampionIndex) => void;
  selectedSpot: SelectedSpot | null;
  onSpotSelected: (index: ChampionIndex, team: TeamIndex) => void;
}

const TeamPanel: React.FC<TeamPanelProps> = ({
  team,
  is_first_team,
  onDeleteChampion,
  selectedSpot,
  onSpotSelected,
}) => {
  const pannelTeamIndex = is_first_team ? 1 : 2;
  const [hoveredChampion, setHoveredChampion] = useState<number | null>(null);

  // Function to handle spot click
  const handleSpotClick = (index: ChampionIndex) => {
    onSpotSelected(index, pannelTeamIndex);
  };

  // Function to handle delete champion click
  const handleDeleteChampion = (
    index: ChampionIndex,
    event: React.MouseEvent
  ) => {
    event.stopPropagation(); // Prevent click event from bubbling up
    onDeleteChampion(index);
    setHoveredChampion(null);
  };

  return (
    <div
      className={clsx("flex flex-col h-full rounded", {
        "bg-blue-900": is_first_team,
        "bg-red-900": !is_first_team,
      })}
    >
      <div className="flex flex-col flex-1">
        <TeamTitle is_blue_side={is_first_team} />
        <ul className="flex flex-col flex-1 justify-between">
          {roles.map((role, index) => (
            <li className="" key={index}>
              <RoleListItem
                key={role}
                role={role}
                index={index as ChampionIndex}
                teamMember={team[index as ChampionIndex]}
                is_first_team={is_first_team}
                selectedSpot={selectedSpot}
                pannelTeamIndex={pannelTeamIndex}
                handleSpotClick={handleSpotClick}
                setHoveredChampion={setHoveredChampion}
                hoveredChampion={hoveredChampion}
                handleDeleteChampion={handleDeleteChampion}
              />
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default TeamPanel;
