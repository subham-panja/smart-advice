'use client';

import { ReactNode } from 'react';
import { useSidebar } from '../contexts/SidebarContext';
import Navbar from './Navbar';

interface MainContentProps {
  children: ReactNode;
}

const MainContent = ({ children }: MainContentProps) => {
  const { isCollapsed } = useSidebar();

  return (
    <div
      className={`flex-1 flex flex-col min-h-screen transition-all duration-300 min-w-0 w-full ${
        isCollapsed ? 'ml-0 md:ml-16' : 'ml-0 md:ml-64'
      }`}
    >
      <Navbar />
      <main className="flex-1 p-3 sm:p-5 lg:p-8 min-w-0 w-full overflow-x-hidden">
        {children}
      </main>
    </div>
  );
};

export default MainContent;
