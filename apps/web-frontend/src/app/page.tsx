import Link from "next/link";
import Image from "next/image";

export default function HomePage() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      <div className="container flex flex-col items-center justify-center gap-8 px-4 py-16 ">
        <h1 className="text-5xl font-extrabold tracking-tight text-primary sm:text-[5rem]">
          Draftking
        </h1>
        <h2 className="text-center text-3xl font-bold">
          Your <span className="text-primary">League of Legends</span> Draft Analyzer
        </h2>

        <div className="mt-6">
          <Link
            href="/draft"
            className="inline-block rounded bg-primary px-6 py-3 text-xl font-bold text-primary-foreground hover:bg-primary/90 transition duration-300"
          >
            <div className="flex items-center justify-center space-x-2">
              <span>Analyze a Draft Now</span>
            </div>
          </Link>
        </div>
        <div className="flex flex-col items-center justify-center gap-8">
          <h3 className="text-2xl font-bold">
            Use Draftking to win your drafts
          </h3>
          <div className="flex flex-col items-center justify-center gap-4 lg:flex-row lg:items-end">
            <div className="flex flex-col items-center">
              <Image
                src="/draft_steps/1.png"
                alt="Step 1 Add champions to Favorites"
                width={450 * 1.2}
                height={338 * 1.2}
              />
              <p className="mt-2 text-center">1. Add champions to Favorites</p>
            </div>
            <div className="flex flex-col items-center">
              <Image
                src="/draft_steps/2.png"
                alt="Step 2 Follow the draft and ask for suggestions"
                width={402 * 1.2}
                height={338 * 1.2}
              />
              <p className="mt-2 text-center">
                2. Follow the draft and ask for suggestions
              </p>
            </div>
            <div className="flex flex-col items-center">
              <Image
                src="/draft_steps/3.png"
                alt="Step 3 Pick the best champion"
                width={380 * 1.2}
                height={338 * 1.2}
              />
              <p className="mt-2 text-center">3. Pick the best champion</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
