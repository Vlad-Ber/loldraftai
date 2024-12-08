import Link from "next/link";
import { ThemeProvider } from "@/components/theme-provider";
import { Inter } from "next/font/google";
import type { Metadata } from "next";
import "./globals.css";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

// Default to the production backend, otherwise vercel builds will fail
const backendUrl =
  process.env.INFERENCE_BACKEND_URL ?? "http://40.67.128.19:8000";

export const metadata: Metadata = {
  title: "Draftking - The Best League of Legends Draft Analysis Tool",
  description:
    "Draftking is a tool for analyzing League of Legends drafts. Use it to select the best champion and win your draft!",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
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
      <body
        className={`font-sans ${inter.variable} min-h-screen bg-background text-foreground`}
      >
        <ThemeProvider>
          <nav className="sticky top-0 z-50 border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <NavigationMenu className="mx-auto px-4 py-3">
              <NavigationMenuList className="flex justify-center gap-10">
                <NavigationMenuItem>
                  <Link href="/" legacyBehavior passHref>
                    <NavigationMenuLink
                      className={navigationMenuTriggerStyle()}
                    >
                      Home
                    </NavigationMenuLink>
                  </Link>
                </NavigationMenuItem>
                <NavigationMenuItem>
                  <Link href="/draft" legacyBehavior passHref>
                    <NavigationMenuLink
                      className={navigationMenuTriggerStyle()}
                    >
                      Draft Analysis
                    </NavigationMenuLink>
                  </Link>
                </NavigationMenuItem>
              </NavigationMenuList>
            </NavigationMenu>
          </nav>

          <main className="flex-1">{children}</main>

          <footer className="border-t border-border/40 bg-card">
            <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
              Last model update: {lastModified} on patch {latestPatch}. Contact
              looyyd on Discord for bug reports or feature requests.
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
