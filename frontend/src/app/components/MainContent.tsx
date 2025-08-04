'use client';

import { ReactNode } from 'react';
import { useSidebar } from '../contexts/SidebarContext';

interface MainContentProps {
  children: ReactNode;
}

const MainContent = ({ children }: MainContentProps) => {
  const { isCollapsed } = useSidebar();

  return (
    <main 
      className={`flex-1 transition-all duration-300 p-4 md:p-8 ${
        isCollapsed ? 'ml-16' : 'ml-64'
      } min-h-screen`}
    >
      {children}
    </main>
  );
};

export default MainContent;
