// /apps/web-frontend/src/app/layout.tsx
import Link from "next/link";
import { ThemeProvider } from "@/components/theme-provider";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Chakra_Petch } from "next/font/google";
import type { Metadata, Viewport } from "next";
import "./globals.css";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
  navigationMenuTriggerStyle,
} from "@draftking/ui/components/ui/navigation-menu";
import { ClarityProvider } from "@/components/clarity-provider";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@draftking/ui/components/ui/dropdown-menu";
import { Menu } from "lucide-react";

const font = Chakra_Petch({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-chakra-petch",
});

// Default to the production backend, otherwise vercel builds will fail
const backendUrl =
  process.env.INFERENCE_BACKEND_URL ??
  "https://leaguedraftv2inference.whiteground-3c896ca8.eastus2.azurecontainerapps.io/";

// Add this near your other constants
const CLARITY_PROJECT_ID = process.env.NEXT_PUBLIC_CLARITY_PROJECT_ID ?? "";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#000000",
};

export const metadata: Metadata = {
  metadataBase: new URL("https://loldraftai.com"),
  title: {
    default: "LoLDraftAI - The Best League of Legends Draft Analysis Tool",
    template: "%s | LoLDraftAI",
  },
  description:
    "LoLDraftAI is an AI-powered League of Legends draft tool that helps you win more games by analyzing team compositions and suggesting the best champions.",
  keywords: [
    "League of Legends",
    "League",
    "LoL",
    "Draft Tool",
    "Champion Select",
    "AI",
    "Machine Learning",
    "Draft Analysis",
    "LoLDraftAI",
  ],
  authors: [{ name: "looyyd" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://loldraftai.com",
    title: "LoLDraftAI - The Best League of Legends Draft Analysis Tool",
    description:
      "AI-powered League of Legends draft analysis tool to win your games",
    siteName: "LoLDraftAI",
    images: [
      {
        url: "https://loldraftai.com/og-image.png",
        width: 1200,
        height: 630,
        alt: "LoLDraftAI - League of Legends Draft Analysis Tool",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "LoLDraftAI - The Best League of Legends Draft Analysis Tool",
    description:
      "AI-powered League of Legends draft analysis tool to win your games",
    images: [
      {
        url: "https://loldraftai.com/og-image.png",
        width: 1200,
        height: 630,
        alt: "LoLDraftAI - League of Legends Draft Analysis Tool",
      },
    ],
  },
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/favicon-16x16.png", sizes: "16x16", type: "image/png" },
      { url: "/favicon-32x32.png", sizes: "32x32", type: "image/png" },
    ],
    apple: [
      { url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
    other: [
      {
        rel: "icon",
        type: "image/png",
        sizes: "192x192",
        url: "/logo192.png",
      },
      {
        rel: "icon",
        type: "image/png",
        sizes: "512x512",
        url: "/logo512.png",
      },
    ],
  },
  alternates: {
    canonical: "https://loldraftai.com",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  let lastModified = "Unknown";
  let latestPatch = "Unknown";

  try {
    // TODO: this is a server component so we call the backend directly, could refactor commong code with metadata/route.ts
    const response = await fetch(`${backendUrl}/metadata`, {
      next: { revalidate: 900 }, // Cache for 15 minutes
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const metadata = await response.json();
    lastModified = new Date(metadata.last_modified).toLocaleDateString(
      "en-US",
      {
        day: "numeric",
        month: "long",
      }
    );
    latestPatch = metadata.patches[metadata.patches.length - 1];
  } catch (error) {
    console.error("Failed to fetch model metadata:", error);
  }

  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${font.variable} font-sans`}>
        <ThemeProvider>
          <ClarityProvider projectId={CLARITY_PROJECT_ID} />
          <div className="flex min-h-screen flex-col bg-background text-foreground">
            <nav className="sticky top-0 z-50 border-b border-border/40 bg-neutral-950">
              {/* Desktop Navigation */}
              <NavigationMenu className="mx-auto hidden px-4 py-3 md:block">
                <NavigationMenuList className="flex justify-center gap-10">
                  <NavigationMenuItem>
                    <Link href="/" legacyBehavior passHref>
                      <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                        Home
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <Link href="/draft" legacyBehavior passHref>
                      <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                        Draft Analysis
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <Link href="/download" legacyBehavior passHref>
                      <NavigationMenuLink className={navigationMenuTriggerStyle()}>
                        Download
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                </NavigationMenuList>
              </NavigationMenu>

              {/* Mobile Navigation */}
              <div className="flex justify-end p-4 md:hidden">
                <DropdownMenu>
                  <DropdownMenuTrigger className="flex items-center">
                    <Menu className="h-6 w-6" />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem asChild>
                      <Link href="/">Home</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href="/draft">Draft Analysis</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem asChild>
                      <Link href="/download">Download</Link>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </nav>

            <main className="flex-1">{children}</main>

            <footer className="border-t border-border/40 bg-neutral-950">
              <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
                <div className="mb-2">
                  Last model update: {lastModified} on patch {latestPatch}.
                  After a new patch, expect a few days of delay before an
                  update. Contact looyyd on Discord for bug reports or feature
                  requests.
                </div>
                <div className="text-xs">
                  LoLDraftAI isn&apos;t endorsed by Riot Games and doesn&apos;t
                  reflect the views or opinions of Riot Games or anyone
                  officially involved in producing or managing Riot Games
                  properties. Riot Games, and all associated properties are
                  trademarks or registered trademarks of Riot Games, Inc.
                </div>
              </div>
            </footer>
          </div>
        </ThemeProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
