
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui-custom/Badge';
import AppLayout from '@/components/layout/AppLayout';
import { useToast } from '@/components/ui/use-toast';
import { 
  RefreshCw, 
  Trash2, 
  DownloadCloud, 
  UploadCloud,
  InfoIcon,
  Moon,
  Sun,
  Github
} from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

const Settings: React.FC = () => {
  const { toast } = useToast();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  const handleClearData = async () => {
    toast({
      title: "Coming Soon",
      description: "Data clearing will be available in a future update",
    });
    setIsDeleteDialogOpen(false);
  };
  
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    
    if (!isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    
    toast({
      title: !isDarkMode ? "Dark Mode Enabled" : "Light Mode Enabled",
      description: "Theme preference saved",
    });
  };
  
  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-medium mb-2">Settings</h1>
        <p className="text-muted-foreground mb-6">Manage your preferences and app data</p>
        
        <div className="space-y-6">
          <div>
            <h2 className="text-xl font-medium mb-2">Appearance</h2>
            <Separator className="mb-4" />
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">Dark Mode</span>
                    <Badge variant="outline" size="sm">Beta</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Switch between light and dark theme
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {isDarkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                  <Switch
                    checked={isDarkMode}
                    onCheckedChange={toggleDarkMode}
                  />
                </div>
              </div>
            </div>
          </div>
          
          <div>
            <h2 className="text-xl font-medium mb-2">Data Management</h2>
            <Separator className="mb-4" />
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">Sync Data</span>
                  <p className="text-sm text-muted-foreground">
                    Manually sync changes with the cloud
                  </p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description: "Cloud sync will be available in a future update",
                    });
                  }}
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>Sync Now</span>
                </Button>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">Export Data</span>
                  <p className="text-sm text-muted-foreground">
                    Export your pantry and shopping list data
                  </p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description: "Data export will be available in a future update",
                    });
                  }}
                >
                  <DownloadCloud className="h-4 w-4" />
                  <span>Export</span>
                </Button>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">Import Data</span>
                  <p className="text-sm text-muted-foreground">
                    Import data from a backup file
                  </p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description: "Data import will be available in a future update",
                    });
                  }}
                >
                  <UploadCloud className="h-4 w-4" />
                  <span>Import</span>
                </Button>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium text-destructive">Clear All Data</span>
                  <p className="text-sm text-muted-foreground">
                    Permanently delete all your data
                  </p>
                </div>
                <Button 
                  variant="destructive" 
                  size="sm"
                  className="gap-2"
                  onClick={() => setIsDeleteDialogOpen(true)}
                >
                  <Trash2 className="h-4 w-4" />
                  <span>Clear</span>
                </Button>
              </div>
            </div>
          </div>
          
          <div>
            <h2 className="text-xl font-medium mb-2">About</h2>
            <Separator className="mb-4" />
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">Version</span>
                  <p className="text-sm text-muted-foreground">
                    Current app version
                  </p>
                </div>
                <Badge variant="outline">v1.0.0</Badge>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">Source Code</span>
                  <p className="text-sm text-muted-foreground">
                    View the source code on GitHub
                  </p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description: "GitHub repository will be available soon",
                    });
                  }}
                >
                  <Github className="h-4 w-4" />
                  <span>GitHub</span>
                </Button>
              </div>
              
              <div className="mt-4 p-4 bg-primary/5 rounded-lg border">
                <div className="flex gap-2">
                  <InfoIcon className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-muted-foreground">
                      This app works completely offline. All your data is stored locally 
                      on your device using IndexedDB.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
      
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete all your 
              pantry items, shopping lists, and settings.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleClearData}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete Everything
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppLayout>
  );
};

export default Settings;
