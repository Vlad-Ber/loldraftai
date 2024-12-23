"use client";

import React from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { Download } from "lucide-react";

export default function DownloadPage() {
  const handleDownload = () => {
    window.location.href =
      "https://pub-4984363368204f3bafa83fbe79e22c38.r2.dev/latest/DraftKing.Setup.exe";
  };

  return (
    <main className="flex min-h-[calc(100vh-200px)] flex-col items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-8">
        <h1 className="text-4xl font-bold">Download DraftKing</h1>
        <p className="text-lg text-muted-foreground">
          Download the latest version of DraftKing for Windows to analyze your
          League of Legends games with live game tracking.
        </p>
        <Button size="lg" onClick={handleDownload} className="gap-2">
          <Download className="h-5 w-5" />
          Download for Windows
        </Button>
      </div>
    </main>
  );
}
