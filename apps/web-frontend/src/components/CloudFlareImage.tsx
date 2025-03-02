import Image, { ImageProps } from "next/image";

const CloudFlareImage = (props: ImageProps) => {
  // Construct the full URL directly
  const fullSrc = `https://media.loldraftai.com${props.src}`;

  return (
    <Image
      {...props}
      src={fullSrc}
      alt={props.alt || "Image"} // Ensure alt is always provided
      unoptimized={true} // Let Cloudflare handle optimization
    />
  );
};

export default CloudFlareImage;
