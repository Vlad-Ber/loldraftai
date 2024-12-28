"use client";

import React from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { Download } from "lucide-react";
import { FaWindows } from "react-icons/fa";

export default function DownloadPage() {
  return (
    <main className="flex min-h-[calc(100vh-200px)] flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-8">
        <h1 className="text-4xl font-bold flex items-center justify-center gap-2">
          Download <span className="brand-text">LoLDraftAI</span>{" "}
          <FaWindows className="h-7 w-7" />
        </h1>
        <p className="text-lg text-muted-foreground">
          Download the latest version of{" "}
          <span className="brand-text">LoLDraftAI</span> for Windows to analyze
          your League of Legends games with{" "}
          <span className="inline-flex font-bold items-center">
            live{" "}
            <span className="relative flex h-2 w-2 ml-1">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500"></span>
            </span>
          </span>{" "}
          game tracking.
        </p>
        <Button
          asChild
          size="lg"
          className="gap-2"
        >
          <a
            href="https://releases.draftking.lol/latest/LoLDraftAI.Setup.exe"
            download
          >
            <Download className="h-5 w-5" />
            Download for Windows
          </a>
        </Button>
      </div>
    </main>
  );
}
