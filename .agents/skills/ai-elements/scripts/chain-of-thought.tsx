"use client";

import {
  ChainOfThought,
  ChainOfThoughtContent,
  ChainOfThoughtHeader,
  ChainOfThoughtImage,
  ChainOfThoughtSearchResult,
  ChainOfThoughtSearchResults,
  ChainOfThoughtStep,
} from "@/components/ai-elements/chain-of-thought";
import { Image } from "@/components/ai-elements/image";
import { ImageIcon, SearchIcon } from "lucide-react";

const exampleImage = {
  base64: "a very large base64 string",
  mediaType: "image/jpeg",
  uint8Array: new Uint8Array([]),
};

const ChainOfThoughtExample = () => (
  <ChainOfThought defaultOpen>
    <ChainOfThoughtHeader />
    <ChainOfThoughtContent>
      <ChainOfThoughtStep
        icon={SearchIcon}
        label="Searching for profiles for Hayden Bleasel"
        status="complete">
        <ChainOfThoughtSearchResults>
          {[
            "https://www.x.com",
            "https://www.instagram.com",
            "https://www.github.com",
          ].map((website) => (
            <ChainOfThoughtSearchResult key={website}>
              {new URL(website).hostname}
            </ChainOfThoughtSearchResult>
          ))}
        </ChainOfThoughtSearchResults>
      </ChainOfThoughtStep>

      <ChainOfThoughtStep
        icon={ImageIcon}
        label="Found the profile photo for Hayden Bleasel"
        status="complete">
        <ChainOfThoughtImage caption="Hayden Bleasel's profile photo from x.com, showing a Ghibli-style man.">
          <Image
            {...exampleImage}
            alt="Example generated image"
            className="aspect-square h-[150px] border"
          />
        </ChainOfThoughtImage>
      </ChainOfThoughtStep>

      <ChainOfThoughtStep
        label="Hayden Bleasel is an Australian product designer, software engineer, and founder. He is currently based in the United States working for Vercel, an American cloud application company."
        status="complete"
      />

      <ChainOfThoughtStep
        icon={SearchIcon}
        label="Searching for recent work..."
        status="active">
        <ChainOfThoughtSearchResults>
          {["https://www.github.com", "https://www.dribbble.com"].map(
            (website) => (
              <ChainOfThoughtSearchResult key={website}>
                {new URL(website).hostname}
              </ChainOfThoughtSearchResult>
            ),
          )}
        </ChainOfThoughtSearchResults>
      </ChainOfThoughtStep>
    </ChainOfThoughtContent>
  </ChainOfThought>
);

export default ChainOfThoughtExample;
