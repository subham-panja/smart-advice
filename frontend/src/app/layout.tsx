import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import Sidebar from "./components/Sidebar"
import MainContent from "./components/MainContent"
import { ThemeProvider } from "./contexts/ThemeContext"
import { SidebarProvider } from "./contexts/SidebarContext"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Stock Advice Dashboard",
  description: "AI-powered stock analysis and recommendations",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors`}
        suppressHydrationWarning={true}
      >
        <ThemeProvider>
          <SidebarProvider>
            <div className="min-h-screen flex">
              <Sidebar />
              <MainContent>
                {children}
              </MainContent>
            </div>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}
