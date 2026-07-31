import type { MetadataRoute } from "next";

// Web app manifest: the Android/Chrome half of the standalone launch that
// apple-mobile-web-app-capable buys on iOS (see app/layout.tsx). Colours track
// --bg-page from globals.css.
export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Sheepshead AI",
    short_name: "Sheepshead",
    description: "Play Sheepshead vs AI or friends",
    start_url: "/",
    display: "standalone",
    // No orientation lock: the table has a real desktop layout above 768px, so
    // an install on a tablet should still be free to turn.
    background_color: "#f3f1ea",
    theme_color: "#f3f1ea",
    icons: [
      { src: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icon-512.png", sizes: "512x512", type: "image/png" },
      {
        src: "/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
  };
}
