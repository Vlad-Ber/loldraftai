import type { ImageComponent } from "@draftking/ui/lib/types";

// Plain image component for Electron
export const PlainImage: ImageComponent = ({ src, ...props }) => {
  // In production, the paths need to be relative to the dist directory
  const imagePath =
    window.location.protocol === "file:"
      ? src.replace("/icons/", "./icons/") // Convert absolute path to relative
      : src;

  return <img src={imagePath} {...props} />;
};
