"use client";

import { Image } from "@/components/ai-elements/image";

const exampleImage = {
  base64: "a very large base64 string",
  mediaType: "image/jpeg",
  uint8Array: new Uint8Array([]),
};

const Example = () => (
  <Image
    {...exampleImage}
    alt="Example generated image"
    className="aspect-square h-[150px] border"
  />
);

export default Example;
