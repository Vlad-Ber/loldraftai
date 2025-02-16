"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";

export default function ChampionRecommendationShowcase() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      {/* Header Section */}
      <div className="w-full bg-gradient-to-b from-primary/10 to-background py-8">
        <div className="container flex flex-col items-center justify-center gap-4 px-4">
          <h1 className="text-4xl font-bold tracking-tight text-primary text-center">
            How to Use Champion Recommendations in LoLDraftAI
          </h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <time dateTime="2024-03-21">March 21, 2024</time>
          </div>
        </div>
      </div>

      {/* Article Content */}
      <article className="container px-4 py-12 prose prose-invert max-w-3xl">
        <p>
          LoLDraftAI&apos;s champion recommendation system is a powerful tool
          that can help you make better champion selections during draft. In
          this post, we&apos;ll look at a real game that showcases how to
          effectively use these recommendations, and discuss the advantages of
          using our desktop application for live draft tracking.
        </p>
        <h2>Desktop App Advantages: Live Draft Tracking</h2>
        <p>
          While the web version is powerful, the{" "}
          <Link href="/download">desktop application</Link> offers some
          significant advantages:
        </p>
        <ul>
          <li>Automatic tracking of picks and bans in real-time</li>
          <li>Instantly updates recommendations as champions are selected</li>
          <li>Automatically greys out picked or banned champions</li>
        </ul>
        <div className="note bg-secondary/10 p-4 rounded-lg my-6">
          <p className="text-sm">
            Note: The games shown were played on patch 15.03 while using a model
            trained on patch 15.02. This demonstrates an important point:
            LoLDraftAI remains highly effective even when slightly behind the
            current patch, as meta shifts are minimal from patch to patch.
          </p>
        </div>
        <h2>Game Showcase: Perfect Taric Counter-Pick</h2>
        <p>
          In our showcase game, we&apos;ll look at a textbook example of using
          champion recommendations to counter the enemy team composition. The
          enemy team selected Camille, Jarvan IV, and Galio - a dive
          composition, which was totally counter-picked by Taric.
        </p>
        <p>
          You can see below the recording of how to use LoLDraftAI champion
          suggestion in a live game.
        </p>
        <div className="flex justify-center">
          <iframe
            width="560"
            height="315"
            src="https://www.youtube.com/embed/0VoN0DCACzE?si=ix3HwyauBjrEjIq4"
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          ></iframe>
        </div>
        <p>
          It&apos;s often useful to start checking suggestions early in the
          draft. This allows you to:
        </p>
        <ul>
          <li>Plan potential picks in advance</li>
          <li>
            Communicate options with your team(through prepicking in champ
            select)
          </li>
        </ul>
        <p>
          I ended up picking in R4(red fourth pick). We could already see the
          Galio/Jarvan combination at this point of the draft so I was quite
          confident the Taric pick would work out well
        </p>
        <p>
          In the final draft, Taric ended up being an excellent counter-pick for
          several reasons:
        </p>
        <ul>
          <li>Strong counter to the enemy dive composition</li>
          <li>
            Excellent synergy with our team&apos;s with taric E being a good
            chain CC with Gragas, Xin Zhao, Yasuo
          </li>
          <li>
            Good lane matchup, where we can just farm and outscale the ennemy
            botlane
          </li>
        </ul>
        Here is the final draft analysis for the game, you can see Taric remains
        a great pick with 9% winrate impact(meaning without Taric, the team
        would on average have 9% less winrate):
        <ClickableImage
          src="/blog/champion-recommendation-showcase/game 1 final analysis.png"
          alt="Draft analysis showing Taric's high impact"
          width={800}
          height={450}
          className="my-6 rounded-lg"
        />
        The analysis was totally correct in this game, and we won convincingly
        with Taric being a really good pick especially against their topside
        dive.{" "}
        <a
          href="https://www.op.gg/summoners/euw/LoLDraftAI-loyd/matches/TtNFybHTVlUkyADsLJL5GfiP8aOx0Lkez9BTPg94f7A%3D/1739121021000"
          target="_blank"
          rel="noopener noreferrer"
        >
          See game results.
        </a>
        <h2>Additional tips</h2>
        Here are two additional tips for using champion recommendations.
        <h3>Tip 1: Picking later is better</h3>
        If you have a deep champion pool, you should as to pick later in the
        draft. The model trully shines when both team comps are mostly known,
        because it can help quickly find picks that totally counter the ennemy
        team.
        <h3>Tip 2: Pick champions that you can play well</h3>
        It may sound obvious, but it&apos;s always better to pick a champion
        that is less recommended but that you can play well. Understand that the
        model is trained on solo queue games where most people pick champs that
        they are good at, so the predictions assume you can play the champion
        well.
        <div className="mt-8 p-4 bg-primary/10 rounded-lg">
          <h3 className="text-xl font-bold mb-2">Ready to Try It Yourself?</h3>
          <p>
            Experience LoLDraftAI&apos;s champion recommendations in your own
            games:
          </p>
          <ul>
            <li>
              Use the <Link href="/draft">web version</Link> for quick analysis
            </li>
            <li>
              Download the <Link href="/download">desktop app</Link> for live
              draft tracking
            </li>
          </ul>
        </div>
      </article>
    </main>
  );
}
