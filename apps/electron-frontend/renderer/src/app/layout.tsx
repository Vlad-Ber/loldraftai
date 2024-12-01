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

const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

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
    // TODO: should have a way to call the vercel backend from the electron app!
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
        className={`font-sans ${inter.variable} min-h-100vh bg-background text-foreground`}
      >
        <ThemeProvider>
          <main className="flex-1 pt-6 px-4">{children}</main>

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
