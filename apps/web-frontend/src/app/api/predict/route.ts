import { NextResponse } from "next/server";

const backendUrl = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const response = await fetch(`${backendUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Prediction error:", error);
    return NextResponse.json(
      { error: "Failed to get prediction" },
      { status: 500 }
    );
  }
}
