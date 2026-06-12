import './globals.css';
import { Instrument_Serif, Geist, JetBrains_Mono } from 'next/font/google';

const instrumentSerif = Instrument_Serif({
  weight: '400',
  style: ['normal', 'italic'],
  subsets: ['latin'],
  variable: '--font-instrument-serif',
  display: 'swap',
});

const geist = Geist({
  subsets: ['latin'],
  variable: '--font-geist',
  display: 'swap',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata = {
  title: 'Sheepshead AI',
  description: 'Play Sheepshead vs AI or friends',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const fontVars = `${instrumentSerif.variable} ${geist.variable} ${jetbrainsMono.variable}`;
  return (
    <html lang="en" className={fontVars}>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
      </head>
      <body>{children}</body>
    </html>
  );
}
