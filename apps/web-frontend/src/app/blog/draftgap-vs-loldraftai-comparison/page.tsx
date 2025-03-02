"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function DraftGapComparison() {
  return (
    <BlogLayout
      title="DraftGap vs LoLDraftAI: A Detailed Comparison"
      date="March 2, 2025"
    >
      <p>
        When it comes to League of Legends draft analysis tools, not all are
        created equal. In this detailed comparison, we'll examine how{" "}
        <span className="brand-text">LoLDraftAI</span> stacks up against{" "}
        <a href="https://draftgap.com">DraftGap</a>, another popular draft
        analysis tool. Through statistical validation, we've found that{" "}
        <span className="brand-text">LoLDraftAI</span> consistently outperforms
        DraftGap in prediction accuracy (65.6% vs 56.5% on unseen data), thanks
        to its more sophisticated understanding of league dynamics. This article
        breaks down the key differences that make{" "}
        <span className="brand-text">LoLDraftAI</span> the superior draft
        analysis tool.
      </p>
      <h2>DraftGap shortcomings</h2>
      <p>
        In their own FAQ section, DraftGap explicitly acknowledges their
        limitations:
      </p>
      <div className="note border border-secondary/30">
        <p className="text-sm">
          <strong>Does DraftGap have any shortcomings?</strong> DraftGap is not
          perfect, and there are several things to keep in mind. The overall
          team comp identity is not taken into account. The synergy of duos
          within a team are used in the calculations, but the tool does not know
          about team comp identity like 'engage' or 'poke'. Damage composition
          is also not used in the calculation (but it is shown, above the team
          winrate), so you need to keep this in mind on your own. These
          shortcomings result from the fact that there is not enough data to
          make a perfect prediction. And we do not want to incorporate opinions
          like 'malphite is an engage champion' into the tool, as using just
          data is the most objective way to make a decision.
        </p>
      </div>
      <p>
        As we'll demonstrate in this article,{" "}
        <span className="brand-text">LoLDraftAI</span> has overcome these
        limitations through its advanced machine learning approach, which can
        identify complex team compositions and their interactions. We will see
        some examples of when the statistical approach of DraftGap falls short.
      </p>
      <h3>DraftGap shortcoming example: full AP draft</h3>
      <p>
        Because DraftGap only uses champion pair statistics, it is totally
        unaware of the draft as a whole. For this reason, it will not understand
        when a draft only has AP Damage. This can be showcased by creating a
        full AP Draft with one team having the following champions from top to
        bot:
      </p>
      <ul>
        <li>Top: Vladimir</li>
        <li>Jungle: Fiddlesticks</li>
        <li>Middle: Kennen</li>
        <li>Bottom: Heimerdinger</li>
        <li>Support: Taric</li>
      </ul>
      <p>
        When inputting this draft into DraftGap, it predicts a 62.62% win
        chance. <span className="brand-text">LoLDraftAI</span> on the other
        hand, understands that this is a full AP Draft, and predicts a win
        chance of 40.4%.
      </p>
      <p>DraftGap prediction:</p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/full-ap-draftgap.png"
        alt="Full AP Draft DraftGap prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        <span className="brand-text">LoLDraftAI</span> prediction:
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/full-ap-loldraftai.png"
        alt="Full AP Draft LoLDraftAI prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        Importantly, this not only impacts analysis but also champion
        suggestions. For example, against this full AP Draft,{" "}
        <span className="brand-text">LoLDraftAI</span> suggests Ornn as the best
        toplane champion. DraftGap, on the other hand, thinks that the team with
        Ornn top against a full AP Draft only has 40% win chance. Obviously Ornn
        would just be unkillable against a full AP Draft, this is a glaring
        example of how DraftGap's statistical approach is limited.
      </p>
      <h3>Shortcomings conclusion</h3>
      <p>
        This full AP draft just serves as a simple illustration, but it also
        will impact more nuanced situations. DraftGap will not understand:
      </p>
      <ul>
        <li>Blue vs Red side differences</li>
        <li>When a team has only one or multiple carries</li>
        <li>When a team has no CC</li>
        <li>When a team has low total damage</li>
        <li>When a team only has late game champions</li>
      </ul>
      <p>
        When you add up all these small subtleties, this just makes DraftGap not
        a very accurate tool, and this is what we will see in the next section
        that compares the accuracy of DraftGap to{" "}
        <span className="brand-text">LoLDraftAI</span>.
      </p>
      <h2>Statistical accuracy comparison</h2>
      <p>
        I have assembled a dataset of 5000 games from patch 15.4 to compare the
        accuracy of DraftGap and <span className="brand-text">LoLDraftAI</span>.
      </p>
      <h3>Dataset</h3>
      <p>
        The dataset consists of 5000 games from patch 15.4. The games are from
        EUW ranked solo queue. These are only randomly sampled games that the{" "}
        <span className="brand-text">LoLDraftAI</span> model has not seen during
        training. The dataset and full results can be found in this{" "}
        <a href="https://docs.google.com/spreadsheets/d/1D7I98rvveX-msgeGkwDBcLhVE2YWhvR4V1SI6Ky5kko/edit?usp=sharing">
          google sheet.
        </a>
      </p>
      <h3>Results</h3>
      <p>
        Note: When calculating the accuracy, a correct guess is when the side
        that actually won was predicted to have more than 50% chance of winning.
      </p>
      <p>Overall accuracy:</p>
      <ul>
        <li>DraftGap: 56.46% (2823/5000 correct)</li>
        <li>
          <span className="brand-text">LoLDraftAI</span>: 65.56% (3278/5000
          correct)
        </li>
      </ul>
      <p>Disagreement between models:</p>
      <ul>
        <li>Model Agreement: 3233/5000 samples (64.66%)</li>
        <li>Model Disagreement: 1767/5000 samples (35.34%)</li>
        <li>Accuracy when models agree: 67.03%</li>
        <li>When models disagree, DraftGap accuracy: 37.13%</li>
        <li>
          When models disagree, <span className="brand-text">LoLDraftAI</span>{" "}
          accuracy: 62.87%
        </li>
      </ul>
      <p>
        These results demonstrate that{" "}
        <span className="brand-text">LoLDraftAI</span> is significantly more
        accurate than DraftGap, correctly predicting 65.56% of the games. While
        it is still impressive that with only a statistical approach, DraftGap
        can predict 56.46% of the games, it is clear that{" "}
        <span className="brand-text">LoLDraftAI</span> is the superior tool.
      </p>
      <div className="mt-8 p-4 bg-primary/10 rounded-lg">
        <h2 className="text-xl font-bold mb-2">Try LoLDraftAI Today</h2>
        <p>
          Experience the advanced draft analysis capabilities of{" "}
          <span className="brand-text">LoLDraftAI</span> yourself:
        </p>
        <ul>
          <li>
            Use the <Link href="/draft">web version</Link> for analysis
          </li>
          <li>
            Download the <Link href="/download">desktop app</Link> for live
            draft tracking
          </li>
        </ul>
      </div>
      <h2>Appendix A: Dataset Verification</h2>
      <p>
        The results for DraftGap were obtained by using their source code
        available here:{" "}
        <a href="https://github.com/vigovlugt/draftgap">
          https://github.com/vigovlugt/draftgap
        </a>
        . All results can be manually verified by using their websites and the
        match id. Example verification for the first match of the dataset: Match
        results:{" "}
        <a href="https://www.leagueofgraphs.com/match/EUW/7298239127">
          https://www.leagueofgraphs.com/match/EUW/7298239127
        </a>
        <br />
      </p>
      <p>DraftGap prediction:</p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/verification-draftgap.png"
        alt="DraftGap prediction verification"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        <span className="brand-text">LoLDraftAI</span> prediction:
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/verification-loldraftai.png"
        alt="LoLDraftAI prediction verification"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <h2>Appendix B: Head to Head Comparison</h2>
      <p>
        It can be a fun exercise to make the 2 draft tools compete head to head
        in a draft. I went through this exercise twice, but didn't include it in
        the main part of the article because it doesn't showcase the differences
        in such a clear cut manner as the dataset comparison. Both these drafts
        were obtained by having one tool select champions for one team and the
        other tool for the other team. The champion with the highest predicted
        winrate was always picked no matter the role, in the standard
        competitive pick order(blue first pick, 2 picks red, 2 pick blue etc.).
      </p>
      <h3>First draft result</h3>
      <p>Blue side by DraftGap</p>
      <ul>
        <li>Top: Teemo</li>
        <li>Jungle: Yorick</li>
        <li>Middle: Kennen</li>
        <li>Bottom: Nilah</li>
        <li>Support: Taric</li>
      </ul>
      <p>
        Red side by <span className="brand-text">LoLDraftAI</span>
      </p>
      <ul>
        <li>Top: Mundo</li>
        <li>Jungle: Sejuani</li>
        <li>Middle: Cho'Gath</li>
        <li>Bottom: Karthus</li>
        <li>Support: Brand</li>
      </ul>
      <p>
        <span className="brand-text">LoLDraftAI</span> predicts an 80% winrate
        for red side, while DraftGap predicts a 63% winrate for blue side.
      </p>
      <p>
        While not as clear cut as the full AP example, I think this is another
        example of how <span className="brand-text">LoLDraftAI</span>{" "}
        outperforms DraftGap. I think red side has comfortable lanes, especially
        with 2 tanky solo lanes that will be able to rush Magic Resist and be
        unkillable in lane. The solo lanes are made even harder for blue side by
        the presence of Karthus ult and ganks from Sejuani.
        <br />
        <br />
        In my opinion this is also a showcase of how DraftGap can make picks
        that make sense as pairs, but don't make sense as a whole. In contrast,{" "}
        <span className="brand-text">LoLDraftAI</span> has crafted an original
        draft that has a lot of tanks but still enough damage to kill the
        squishy enemy team and where it doesn't matter if the backline of
        Brand/Karthus is focused, because they can deal their damage no matter
        what.
      </p>
      <p>DraftGap prediction:</p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/head-to-head-1-draftgap.png"
        alt="Head to head draft 1 DraftGap prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        <span className="brand-text">LoLDraftAI</span> prediction:
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/head-to-head-1-loldraftai.png"
        alt="Head to head draft 1 LoLDraftAI prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <h3>Second draft result</h3>
      <p>
        Blue side by <span className="brand-text">LoLDraftAI</span>
      </p>
      <ul>
        <li>Top: Vayne</li>
        <li>Jungle: Nunu</li>
        <li>Middle: Cho'Gath</li>
        <li>Bottom: Sivir</li>
        <li>Support: Pyke</li>
      </ul>
      <p>Red side by DraftGap</p>
      <ul>
        <li>Top: Malphite</li>
        <li>Jungle: Yorick</li>
        <li>Middle: Kayle</li>
        <li>Bottom: Nilah</li>
        <li>Support: Nami</li>
      </ul>
      <p>
        <span className="brand-text">LoLDraftAI</span> predicts a 65% winrate
        for blue side, while DraftGap predicts a 67% winrate for red side.
      </p>
      <p>
        Here again, the models disagree. But I think this reveals another
        limitation of DraftGap, it is not aware of early/late game dynamics. In
        this game, I think it is quite easy for the blue team to snowball hard
        with a pushing mid lane and early gank pressure from Pyke/Nunu. And if
        they manage to snowball and feed either Vayne/Sivir, they can probably
        close the game out quickly. This will be even easier because of the
        objective secures granted by Nunu/Cho'Gath.
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/head-to-head-2-draftgap.png"
        alt="Head to head draft 2 DraftGap prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/head-to-head-2-loldraftai.png"
        alt="Head to head draft 2 LoLDraftAI prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <h3>Conclusion</h3>
      <p>
        The head to head comparison, while not as clear cut as the examples in
        the main article, still can be interpreted as a showcase of how{" "}
        <span className="brand-text">LoLDraftAI</span> is able to understand
        more nuanced dynamics, rather than just statistical pairings.
      </p>
    </BlogLayout>
  );
}
