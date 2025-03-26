
import React from 'react';
import { Outlet } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import AppNavigation from './AppNavigation';
import { cn } from '@/lib/utils';

interface AppLayoutProps {
  children?: React.ReactNode;
  hideNavigation?: boolean;
  className?: string;
}

const AppLayout: React.FC<AppLayoutProps> = ({ 
  children, 
  hideNavigation = false,
  className 
}) => {
  return (
    <div className="min-h-screen flex flex-col bg-background relative">
      <main 
        className={cn(
          "flex-1 container max-w-screen-lg mx-auto px-4 sm:px-6 pb-32 pt-4", 
          hideNavigation ? "pb-4" : "pb-32",
          className
        )}
      >
        <AnimatePresence mode="wait">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="w-full"
          >
            {children || <Outlet />}
          </motion.div>
        </AnimatePresence>
      </main>

      {!hideNavigation && <AppNavigation />}
    </div>
  );
};

export default AppLayout;
