import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "五子棋 AI 训练平台",
  description: "Gomoku AI Training Platform - 基于 AlphaZero 算法的五子棋人工智能训练与对弈系统",
  keywords: ["Gomoku", "五子棋", "AI", "AlphaZero", "MCTS", "训练"],
  authors: [{ name: "Gomoku AI Team" }],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "五子棋 AI 训练平台",
    description: "基于 AlphaZero 算法的五子棋人工智能训练与对弈系统",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "五子棋 AI 训练平台",
    description: "基于 AlphaZero 算法的五子棋人工智能训练与对弈系统",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
