import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  // React Compiler runs a Babel pass on every module in dev and can exhaust
  // the Node heap on large component trees (ai-elements, prompt-input, etc.).
  reactCompiler: !isDev,
  experimental: {
    optimizePackageImports: [
      "lucide-react",
      "@hugeicons/react",
      "@hugeicons/core-free-icons",
    ],
  },
  // Evict compiled pages sooner so the dev server does not retain every route.
  onDemandEntries: {
    maxInactiveAge: 60 * 1000,
    pagesBufferLength: 2,
  },
};

export default nextConfig;
