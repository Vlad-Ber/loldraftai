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

export const metadata: Metadata = {
  title: "Draftking - The Best League of Legends Draft Analysis Tool",
  description:
    "Draftking is a tool for analyzing League of Legends drafts. Use it to select the best champion and win your draft!",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
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
              Last model update: 14 March on patch 14.5. Contact looyyd on
              Discord for bug reports or feature requests.
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
