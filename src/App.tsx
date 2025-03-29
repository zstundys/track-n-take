import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import AddItem from "./pages/AddItem";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Settings from "./pages/Settings";
import ShoppingList from "./pages/ShoppingList";

// Import i18n configuration
import "./i18n";

const queryClient = new QueryClient();

const App = () => {
  // usePwaServiceWorker();
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter basename="/track-n-take/">
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/shopping-list" element={<ShoppingList />} />
            <Route path="/add-item" element={<AddItem />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
