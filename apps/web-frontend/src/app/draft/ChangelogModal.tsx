import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@draftking/ui/components/ui/dialog";
import { InfoIcon } from "lucide-react";

export interface ChangelogModalProps {
  isOpen: boolean;
  closeHandler: () => void;
}

export const ChangelogModal: React.FC<ChangelogModalProps> = ({
  isOpen,
  closeHandler,
}) => {
  return (
    <Dialog open={isOpen} onOpenChange={closeHandler}>
      <DialogContent className="max-h-[80vh] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <InfoIcon className="h-5 w-5" />
            Important Announcement
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm leading-relaxed">
              This will be the last patch update for the project. While all
              features will remain available, the model will not be updated for
              future patches.
            </p>
            <p className="text-sm leading-relaxed">
              Thank you for your support and interest in this project. The
              decision to pause updates was made after careful consideration of
              the project&apos;s long-term sustainability.
            </p>
            <p className="text-sm leading-relaxed">
              Furthermore, the source code for the repo has been made totally
              open source:{" "}
              <a
                href="https://github.com/Looyyd/loldraftai-monorepo"
                className="text-blue-500"
              >
                https://github.com/Looyyd/loldraftai-monorepo
              </a>
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
