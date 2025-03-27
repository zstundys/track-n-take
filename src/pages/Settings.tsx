import React, { useState } from "react";
import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui-custom/Badge";
import AppLayout from "@/components/layout/AppLayout";
import { useToast } from "@/components/ui/use-toast";
import {
  RefreshCw,
  Trash2,
  DownloadCloud,
  UploadCloud,
  InfoIcon,
  Moon,
  Sun,
  Github,
} from "lucide-react";
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
import LanguageSelector from "@/components/language-selector";

const Settings: React.FC = () => {
  const { toast } = useToast();
  const { t } = useTranslation();
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
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
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
        <h1 className="text-3xl font-medium mb-2">{t("settings.title")}</h1>
        <p className="text-muted-foreground mb-6">{t("settings.subtitle")}</p>

        <div className="space-y-6">
          <div>
            <h2 className="text-xl font-medium mb-2">
              {t("settings.appearance.title")}
            </h2>
            <Separator className="mb-4" />

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">
                      {t("settings.appearance.darkMode")}
                    </span>
                    <Badge variant="outline" size="sm">
                      Beta
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.appearance.darkModeDescription")}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {isDarkMode ? (
                    <Moon className="h-4 w-4" />
                  ) : (
                    <Sun className="h-4 w-4" />
                  )}
                  <Switch
                    checked={isDarkMode}
                    onCheckedChange={toggleDarkMode}
                  />
                </div>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-medium mb-2">
              {t("settings.language.title")}
            </h2>
            <Separator className="mb-4" />

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.language.title")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.language.description")}
                  </p>
                </div>
                <div className="w-[200px]">
                  <LanguageSelector />
                </div>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-medium mb-2">
              {t("settings.dataManagement.title")}
            </h2>
            <Separator className="mb-4" />

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.dataManagement.syncData")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.dataManagement.syncDescription")}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description:
                        "Cloud sync will be available in a future update",
                    });
                  }}
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>{t("settings.dataManagement.syncButton")}</span>
                </Button>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.dataManagement.exportData")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.dataManagement.exportDescription")}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description:
                        "Data export will be available in a future update",
                    });
                  }}
                >
                  <DownloadCloud className="h-4 w-4" />
                  <span>{t("settings.dataManagement.exportButton")}</span>
                </Button>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.dataManagement.importData")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.dataManagement.importDescription")}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description:
                        "Data import will be available in a future update",
                    });
                  }}
                >
                  <UploadCloud className="h-4 w-4" />
                  <span>{t("settings.dataManagement.importButton")}</span>
                </Button>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium text-destructive">
                    {t("settings.dataManagement.clearData")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.dataManagement.clearDescription")}
                  </p>
                </div>
                <Button
                  variant="destructive"
                  size="sm"
                  className="gap-2"
                  onClick={() => setIsDeleteDialogOpen(true)}
                >
                  <Trash2 className="h-4 w-4" />
                  <span>{t("settings.dataManagement.clearButton")}</span>
                </Button>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-xl font-medium mb-2">
              {t("settings.about.title")}
            </h2>
            <Separator className="mb-4" />

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.about.version")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.about.versionDescription")}
                  </p>
                </div>
                <Badge variant="outline">v1.0.0</Badge>
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <span className="text-sm font-medium">
                    {t("settings.about.sourceCode")}
                  </span>
                  <p className="text-sm text-muted-foreground">
                    {t("settings.about.sourceDescription")}
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
                  <span>{t("settings.about.sourceButton")}</span>
                </Button>
              </div>

              <div className="mt-4 p-4 bg-primary/5 rounded-lg border">
                <div className="flex gap-2">
                  <InfoIcon className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm text-muted-foreground">
                      {t("settings.about.offlineInfo")}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      <AlertDialog
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("settings.deleteDialog.title")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("settings.deleteDialog.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t("settings.deleteDialog.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleClearData}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("settings.deleteDialog.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppLayout>
  );
};

export default Settings;
