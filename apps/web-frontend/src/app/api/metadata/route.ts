import { NextResponse } from "next/server";

const backendUrl = process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${backendUrl}/metadata`, {
      next: {
        revalidate: 900, // Cache for 15 minutes (900 seconds)
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: {
        "Cache-Control":
          "public, max-age=900, s-maxage=900, stale-while-revalidate=86400",
      },
    });
  } catch (error) {
    console.error("Metadata fetch error:", error);
    return NextResponse.json(
      { error: "Failed to get metadata" },
      { status: 500 }
    );
  }
}
