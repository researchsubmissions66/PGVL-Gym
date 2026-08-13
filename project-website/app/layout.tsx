import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const outfit = Outfit({ variable: "--font-outfit", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;

  return {
    metadataBase: new URL(origin),
    title: "PGVL-Gym | Generalized Pathology VLM Benchmarking",
    description: "A configuration-first framework for fair, reproducible evaluation of whole-slide pathology vision-language models.",
    icons: { icon: "/logo-gym.png", shortcut: "/logo-gym.png" },
    openGraph: {
      title: "PGVL-Gym",
      description: "Fair, reproducible pathology VLM benchmarking.",
      url: origin,
      siteName: "PGVL-Gym",
      images: [{ url: `${origin}/og.png`, width: 1536, height: 1024, alt: "PGVL-Gym — fair, reproducible pathology VLM benchmarking" }],
      type: "website",
    },
    twitter: {
      card: "summary_large_image",
      title: "PGVL-Gym",
      description: "Fair, reproducible pathology VLM benchmarking.",
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${inter.variable} ${outfit.variable}`}>{children}</body></html>;
}
