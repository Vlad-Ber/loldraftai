import Link from "next/link";
import Image from "next/image";

export default function NotFound() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-gradient-to-b from-gray-700 to-gray-900 text-white">
      <Image
        src="/icons/confused_blitz.webp"
        alt="Confused Blitzcrank"
        width={400}
        height={400}
      />
      <h1 className="text-4xl font-bold">404 Not Found</h1>
      <p className="text-xl">Could not find requested resource</p>
      <Link
        href="/draft"
        className="mt-4 inline-block rounded bg-blue-500 px-4 py-2 text-xl font-bold text-white transition duration-300 hover:bg-blue-700"
      >
        Return to draft analysis
      </Link>
    </main>
  );
}
