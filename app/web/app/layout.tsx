import "./globals.css";
import { Instrument_Serif, Geist, JetBrains_Mono } from "next/font/google";

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

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata = {
  title: "Sheepshead AI",
  description: "Play Sheepshead vs AI or friends",
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
}: {
  children: React.ReactNode;
}) {
  const fontVars = `${instrumentSerif.variable} ${geist.variable} ${jetbrainsMono.variable}`;
  return (
    <html lang="en" className={fontVars}>
      <body>{children}</body>
    </html>
  );
}
