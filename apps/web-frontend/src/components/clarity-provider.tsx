// /apps/web-frontend/src/components/clarity-provider.tsx
"use client";

import { useEffect } from "react";
import clarity from "@microsoft/clarity";

export function ClarityProvider({ projectId }: { projectId: string }) {
  useEffect(() => {
    // Only initialize Clarity in production
    if (projectId && process.env.NODE_ENV === "production") {
      clarity.init(projectId);
    }
  }, [projectId]);

  return null;
}
