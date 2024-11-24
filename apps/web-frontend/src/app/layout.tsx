import Link from "next/link";
import { Inter } from "next/font/google";
import type { Metadata } from "next";
import "./globals.css";

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
    <html lang="en">
      <body className={`font-sans ${inter.variable}`}>
        <nav className="bg-gray-800 p-4 text-white">
          <ul className="flex justify-center gap-10">
            <li>
              <Link href="/" className="hover:text-gray-400">
                Home
              </Link>
            </li>
            <li>
              <Link href="/draft" className="hover:text-gray-400">
                Draft Analysis
              </Link>
            </li>
          </ul>
        </nav>
        {children}
        <div className="bg-gray-800 p-4 text-center text-sm text-white">
          Last model update: 14 March on patch 14.5. Contact looyyd on Discord
          for bug reports or feature requests.
        </div>
      </body>
    </html>
  );
}
