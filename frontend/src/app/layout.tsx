import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SafeRx Analytics - Drug Safety Knowledge Assistant",
  description:
    "AI-powered drug safety assistant for pharmacovigilance analysts",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <div className="min-h-screen flex flex-col">
          {/* Header */}
          <header className="bg-primary-700 text-white shadow-md">
            <div className="max-w-7xl mx-auto px-4 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center">
                    <span className="text-primary-700 font-bold text-xl">Rx</span>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold">SafeRx Analytics</h1>
                    <p className="text-primary-200 text-sm">
                      Drug Safety Knowledge Assistant
                    </p>
                  </div>
                </div>
                <nav className="hidden md:flex space-x-6">
                  <a href="/" className="hover:text-primary-200 transition">
                    Chat
                  </a>
                  <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="hover:text-primary-200 transition"
                  >
                    GitHub
                  </a>
                </nav>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1">{children}</main>

          {/* Footer */}
          <footer className="bg-gray-100 border-t">
            <div className="max-w-7xl mx-auto px-4 py-4">
              <p className="text-center text-sm text-gray-500">
                This tool is for informational purposes only. Always consult a
                healthcare professional for medical advice.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
