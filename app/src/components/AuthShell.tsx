import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from './ui/button';
import { LogIn } from 'lucide-react';

interface AuthShellProps {
  children: ReactNode;
}

export function AuthShell({ children }: AuthShellProps) {
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <div className="flex flex-col min-h-screen bg-white">
      {/* Header - minimal black/white design */}
      <header className="fixed top-0 left-0 right-0 bg-white/95 backdrop-blur-sm z-50 border-b border-black/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link to="/" className="flex items-center">
                <img src="/logo.svg" alt="Valthera" className="h-14 w-14" />
              </Link>
            </div>
            <div className="flex items-center gap-3">
              {currentPath !== '/signin' && (
                <Button variant="ghost" size="lg" className="text-black/80 hover:text-black hover:bg-black/5" asChild>
                  <Link to="/signin">
                    <LogIn className="mr-2 h-4 w-4" />
                    Sign In
                  </Link>
                </Button>
              )}
              {currentPath !== '/signup' && (
                <Button size="lg" className="bg-black text-white hover:bg-black/90" asChild>
                  <Link to="/signup">
                    Get Started
                  </Link>
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow flex items-center justify-center pt-16">
        <div className="w-full max-w-md mx-auto px-4 sm:px-6 lg:px-8">
          {children}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-black/10">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex justify-center items-center">
            <p className="text-xs font-mono text-black/40">
              © 2025 Valthera LLC
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
} 