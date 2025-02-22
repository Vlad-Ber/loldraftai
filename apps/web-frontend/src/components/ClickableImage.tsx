"use client";

import CloudFlareImage from "@/components/CloudFlareImage";
import { useState } from "react";

interface ClickableImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  className?: string;
}

export function ClickableImage({
  src,
  alt,
  width,
  height,
  className = "",
}: ClickableImageProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <CloudFlareImage
        src={src}
        alt={alt}
        width={width}
        height={height}
        className={`cursor-pointer ${className}`}
        onClick={() => setIsOpen(true)}
      />
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setIsOpen(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]">
            <CloudFlareImage
              src={src}
              alt={alt}
              width={1920}
              height={1080}
              className="object-contain max-h-[90vh]"
            />
            <button
              className="absolute top-4 right-4 text-white hover:text-primary"
              onClick={() => setIsOpen(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
}
