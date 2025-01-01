import { useDraftStore } from "../stores/draftStore";
import App from "../App";
import { Toaster } from "@draftking/ui/components/ui/toaster";

export function Layout() {
  const { currentPatch } = useDraftStore();
  const lastModified = new Date().toLocaleDateString("en-US", {
    day: "numeric",
    month: "long",
  });

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <main className="flex-1">
        <App />
      </main>

      <footer className="border-t border-border/40 bg-neutral-950">
        <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
          <div className="mb-2">
            Last model update: {lastModified} on patch {currentPatch}. After a
            new patch, expect a few days of delay before an update. Contact
            looyyd on Discord for bug reports or feature requests.
          </div>
          <div className="text-xs">
            LoLDraftAI isn't endorsed by Riot Games and doesn't reflect the
            views or opinions of Riot Games or anyone officially involved in
            producing or managing Riot Games properties. Riot Games, and all
            associated properties are trademarks or registered trademarks of
            Riot Games, Inc.
          </div>
        </div>
      </footer>
      <Toaster />
    </div>
  );
}
