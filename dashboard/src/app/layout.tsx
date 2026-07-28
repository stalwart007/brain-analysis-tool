import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});


export const metadata: Metadata = {
  title: "CogniSwarm",
  description: "Behavioral telemetry → synthetic persona swarms.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: browser extensions (antivirus, form fillers, etc.)
    // inject attributes like `bis_register` / `__processed_<uuid>__` onto <html>/<body>
    // before React hydrates. The server can't know about them, so React would warn.
    // This suppresses the mismatch for these elements' own attributes ONLY (one level
    // deep) — it does not hide real hydration bugs elsewhere in the tree.
    <html
      lang="en"
      className={spaceGrotesk.variable}
      suppressHydrationWarning
    >
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
