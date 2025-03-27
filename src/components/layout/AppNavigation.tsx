import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Home, ShoppingBag, PlusCircle, Settings } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";

const AppNavigation: React.FC = () => {
  const location = useLocation();
  const { t } = useTranslation();

  const navItems = [
    { to: "/", label: t("navigation.pantry"), icon: Home },
    {
      to: "/shopping-list",
      label: t("navigation.shopping"),
      icon: ShoppingBag,
    },
    { to: "/add-item", label: t("navigation.add"), icon: PlusCircle },
    { to: "/settings", label: t("navigation.settings"), icon: Settings },
  ];

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 w-max-content mx-auto">
      <div className="glass p-2 sm:p-3 mx-4 mb-4 rounded-xl sm:rounded-full max-w-sm shadow-lg border border-border dark:border-border/30 dark:bg-background/80 sm:mx-auto">
        <nav className="grid grid-flow-col grid-cols-4 items-center">
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
                    ? "text-primary dark:text-primary-foreground font-medium"
                    : "text-muted-foreground dark:text-muted-foreground/80 hover:text-foreground dark:hover:text-foreground hover:bg-secondary/60 dark:hover:bg-secondary/30"
                )}
              >
                {isActive && (
                  <motion.span
                    layoutId="navIndicator"
                    className="absolute inset-0 bg-primary/10 dark:bg-primary/20 rounded-3xl"
                    initial={false}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}

                <Icon className="h-5 w-5 sm:h-5 sm:w-5" />
                <span className="text-xs mt-1 font-medium">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </div>
  );
};

export default AppNavigation;
