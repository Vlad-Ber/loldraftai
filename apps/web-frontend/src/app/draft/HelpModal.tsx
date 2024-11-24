import React, { useRef, useEffect } from "react";
import { XMarkIcon } from "@heroicons/react/24/solid";
import {
  CursorArrowRaysIcon,
  StarIcon,
  ChartBarIcon,
  PuzzlePieceIcon,
  ArrowsRightLeftIcon,
  TrashIcon,
} from "@heroicons/react/24/outline";

interface HelpModalProps {
  closeHandler: () => void;
}

const HelpModal: React.FC<HelpModalProps> = ({ closeHandler }) => {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Function to detect any mouse clicks happening out of modal i.e., on the overlay
    const handleClickOutside = (event: MouseEvent) => {
      if (
        modalRef.current &&
        !modalRef.current.contains(event.target as Node)
      ) {
        closeHandler();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [closeHandler]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75">
      <div
        ref={modalRef}
        className="relative max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-lg bg-white p-6 text-gray-700 shadow-lg"
      >
        <button
          onClick={closeHandler}
          className="absolute right-0 top-0 mr-4 mt-4 text-gray-500 hover:text-gray-700"
        >
          <XMarkIcon className="h-6 w-6" />
        </button>
        <h2 className="mb-4 text-xl font-bold">How to Use</h2>
        <div className="space-y-4">
          <p className="flex items-center">
            <CursorArrowRaysIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Click on a champion to add them to a team in pick order.
            </span>
          </p>
          <p className="flex items-center">
            <ArrowsRightLeftIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Select a team slot to swap champions or add a new one by clicking
              it.
            </span>
          </p>
          <p className="flex items-center">
            <StarIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Right-click (or long-press on mobile) on a champion to add them to
              your favorites.
            </span>
          </p>
          <p className="flex items-center">
            <TrashIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Right-click (or long-press on mobile) on a champion in team panels
              to remove them.
            </span>
          </p>
          <p className="flex items-center">
            <ChartBarIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Request a draft analysis to see win probabilities and suggestions.
            </span>
          </p>
          <p className="flex items-center">
            <PuzzlePieceIcon className="mr-2 h-6 w-6 flex-shrink-0" />
            <span className="flex-grow">
              Use the suggest champion button and favorites to get champion
              suggestions for a selected slots.
            </span>
          </p>
        </div>
        <p className="mt-6">
          Our machine learning model analyzes various features such as win
          rates, matchups, and team synergies to suggest the best champions for
          your draft.
        </p>
      </div>
    </div>
  );
};

export default HelpModal;
