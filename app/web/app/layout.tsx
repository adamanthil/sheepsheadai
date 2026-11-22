import "./globals.css";
import Analytics from "./components/Analytics";
import { Instrument_Serif, Geist, JetBrains_Mono } from "next/font/google";
import localFont from "next/font/local";

const instrumentSerif = Instrument_Serif({
  weight: "400",
  style: ["normal", "italic"],
  subsets: ["latin"],
  variable: "--font-instrument-serif",
  display: "swap",
});

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
  display: "swap",
});

// The card suits (♠♣♥♦) only: none of the faces above has them, so they
// came from next/font's metric-matched fallbacks, which differ by platform
// (Times New Roman's or Arial's suits on Mac and Windows, something else on
// Android). Sheepshead Suits is a four-glyph subset of Liberation Serif, the
// open Times substitute, so every platform and the link-preview card
// (opengraph-image.tsx) draw the same suits. Each face leads its stack in
// globals.css; the unicode-range keeps it off every other character.
// (next/font needs literal options, so the two calls repeat themselves.)
// The font file is already drawn the way the Instrument Serif fallback drew
// Times New Roman's suits: outlines at 83.94%, Instrument Serif's line
// metrics. So it goes in the display stack as is.
const suitsDisplay = localFont({
  src: "../assets/fonts/SheepsheadSuits-Regular.ttf",
  variable: "--font-suits-display",
  display: "swap",
  declarations: [
    { prop: "unicode-range", value: "U+2660, U+2663, U+2665-2666" },
  ],
  adjustFontFallback: false,
});

// The Geist fallback drew Arial's suits (the same shapes) at 104.76% with
// Geist's line metrics; scaling the file by 104.76 / 83.94 and overriding
// its metrics to Geist's (divided back by that scale) keeps them so.
const suitsUi = localFont({
  src: "../assets/fonts/SheepsheadSuits-Regular.ttf",
  variable: "--font-suits-ui",
  display: "swap",
  declarations: [
    { prop: "unicode-range", value: "U+2660, U+2663, U+2665-2666" },
    { prop: "size-adjust", value: "124.8%" },
    { prop: "ascent-override", value: "80.53%" },
    { prop: "descent-override", value: "23.64%" },
    { prop: "line-gap-override", value: "0%" },
  ],
  adjustFontFallback: false,
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const description =
  "Play live with friends, strangers, and a deep neural network.";

// The deployed site's origin, inlined at build time. CI sets it for the
// production image only (Dockerfile, .github/workflows/ci.yml); local and
// e2e builds leave it unset, so they never claim to be the public site.
const siteUrl = process.env.NEXT_PUBLIC_SITE_URL;

export const metadata = {
  // Link previews need absolute image URLs. Unset, Next resolves
  // opengraph-image against http://localhost:3000, which is right for a
  // build that isn't the public site.
  metadataBase: siteUrl ? new URL(siteUrl) : undefined,
  title: "Sheepshead AI",
  description,
  // The image itself comes from app/opengraph-image.tsx.
  openGraph: {
    type: "website",
    // The canonical URL. Pages that don't set their own inherit the
    // homepage's, so a shared table link previews as the site itself.
    ...(siteUrl && { url: "/" }),
    siteName: "Sheepshead AI",
    title: "Sheepshead AI",
    description,
  },
  twitter: { card: "summary_large_image" },
  manifest: "/manifest.webmanifest",
  // Added to the Home Screen, the app launches without Safari's URL bar and
  // home-indicator chrome, which is worth ~90px of viewport — the table stage
  // is the flexible row that gets all of it, so the ring cards grow with it.
  appleWebApp: {
    capable: true,
    title: "Sheepshead",
    statusBarStyle: "default",
  },
  // `capable` above emits only the modern cross-browser spelling
  // (mobile-web-app-capable); iOS before 17 reads the apple-prefixed name and
  // nothing else, so emit that one too. Newer iOS takes standalone mode from
  // the manifest's display field instead.
  other: { "apple-mobile-web-app-capable": "yes" },
  icons: {
    icon: [
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/apple-touch-icon.png", sizes: "180x180" }],
  },
};

// Disable pinch-zoom so card drag/tap gestures aren't hijacked on mobile.
// viewport-fit stays at its default: iOS then insets the viewport past the
// status bar and home indicator on its own, so a standalone launch needs no
// safe-area padding of its own.
export const viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#f3f1ea",
};

export default function RootLayout({
  children,
  modal,
}: {
  children: React.ReactNode;
  /** The @modal slot: overlays opened by intercepting routes. */
  modal: React.ReactNode;
}) {
  const fontVars = `${suitsDisplay.variable} ${suitsUi.variable} ${instrumentSerif.variable} ${geist.variable} ${jetbrainsMono.variable}`;
  return (
    <html lang="en" className={fontVars}>
      <body>
        {/* Gives iOS 27 a solid colour to sample at the top edge of the
            installed app; see .ios-pwa-status-bar in globals.css. */}
        <div className="ios-pwa-status-bar" aria-hidden="true" />
        {children}
        {modal}
        <Analytics />
      </body>
    </html>
  );
}
