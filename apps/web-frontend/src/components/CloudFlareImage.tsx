import Image, { ImageProps } from "next/image";

const imageLoader = ({ src }: { src: string }) => {
  return `https://media.loldraftai.com${src}`;
};

const CloudFlareImage = (props: ImageProps) => {
  return <Image {...props} loader={imageLoader} />;
};

export default CloudFlareImage;
