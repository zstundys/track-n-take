
import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Home, ShoppingBag, PlusCircle, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';

const AppNavigation: React.FC = () => {
  const location = useLocation();
  
  const navItems = [
    { to: '/', label: 'Pantry', icon: Home },
    { to: '/shopping-list', label: 'Shopping', icon: ShoppingBag },
    { to: '/add-item', label: 'Add', icon: PlusCircle },
    { to: '/settings', label: 'Settings', icon: Settings },
  ];
  
  return (
    <div className="fixed bottom-0 left-0 right-0 z-50">
      <div className="glass p-2 sm:p-3 mx-4 mb-4 rounded-xl sm:rounded-full max-w-sm shadow-lg border sm:mx-auto">
        <nav className="flex justify-around items-center">
          {navItems.map((item) => {
            const isActive = location.pathname === item.to;
            const Icon = item.icon;
            
            return (
              <NavLink 
                key={item.to} 
                to={item.to}
                className={cn(
                  "relative flex flex-col items-center justify-center p-2 sm:p-3 rounded-lg transition-colors",
                  isActive 
                    ? "text-primary" 
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                )}
              >
                {isActive && (
                  <motion.span
                    layoutId="navIndicator"
                    className="absolute inset-0 bg-primary/10 rounded-lg"
                    initial={false}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                
                <Icon className="h-5 w-5 sm:h-5 sm:w-5" />
                <span className="text-xs mt-1">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default AppNavigation;
