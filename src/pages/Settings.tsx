import React, { useState, useEffect } from "react";
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
  Key,
  Loader2,
  CheckCircle,
  XCircle,
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
import { Input } from "@/components/ui/input";
import LanguageSelector from "@/components/language-selector";
import { APP_VERSION, APP_VERSION_DATE } from "@/lib/version";
import { useIntl } from "@/hooks/useIntl";
import { useImageRecognizerWorker } from "@/hooks/use-image-recognizer-worker";
import { assert } from "@/lib/utils";
import { HUGGINGFACE_TOKEN_STORAGE_KEY } from "@/lib/worker-messages";

const THEME_KEY = "pantry-theme-preference";

const Settings: React.FC = () => {
  const { toast } = useToast();
  const { t } = useTranslation();
  const { longDate } = useIntl();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [huggingFaceToken, setHuggingFaceToken] = useState(
    localStorage.getItem(HUGGINGFACE_TOKEN_STORAGE_KEY) || ""
  );

  // Get worker state from hook
  const {
    isWorkerReady,
    isClientValid,
    clear,
    isInitializing,
    validateToken,
    error,
  } = useImageRecognizerWorker();

  console.log("#", { isWorkerReady, isClientValid, isInitializing });

  const { isDarkMode, toggleDarkMode } = useDarkThemePreference();

  const handleClearData = async () => {
    toast({
      title: "Coming Soon",
      description: "Data clearing will be available in a future update",
    });
    setIsDeleteDialogOpen(false);
  };

  const saveHuggingFaceToken = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const isValid = await validateToken(huggingFaceToken);

      assert(isValid, "Invalid token");

      localStorage.setItem(HUGGINGFACE_TOKEN_STORAGE_KEY, huggingFaceToken);

      toast({
        title: "Token Saved & Validated",
        description:
          "Your Hugging Face API token has been saved and validated successfully.",
      });
    } catch {
      localStorage.setItem(HUGGINGFACE_TOKEN_STORAGE_KEY, "");
      toast({
        title: "Invalid Token",
        description:
          "The token couldn't be validated. Please check and try again.",
        variant: "destructive",
      });
    }
  };

  // Render token status icon
  const renderTokenStatus = () => {
    if (isInitializing) {
      return <Loader2 className="h-4 w-4 animate-spin text-primary" />;
    } else if (isClientValid === true) {
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    } else if (isClientValid === false) {
      return <XCircle className="h-4 w-4 text-red-500" />;
    }
    return null;
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

          <form onSubmit={saveHuggingFaceToken}>
            <h2 className="text-xl font-medium mb-2">API Settings</h2>
            <Separator className="mb-4" />

            <div className="space-y-4">
              <div className="flex flex-col md:flex-row md:items-center gap-4 justify-between">
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">
                      Hugging Face API Token
                    </span>
                    <Badge variant="outline" size="sm">
                      Required
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Enter your Hugging Face API token to enable image
                    recognition features
                  </p>
                  {error && <p className="text-sm text-destructive">{error}</p>}
                </div>
                <div className="flex items-center gap-2 w-full md:w-auto">
                  <div className="relative flex-shrink">
                    <Input
                      type="password"
                      placeholder="hf_..."
                      value={huggingFaceToken}
                      onChange={(e) => {
                        clear();
                        return setHuggingFaceToken(e.target.value);
                      }}
                      className="w-full font-mono pr-8"
                    />
                    {huggingFaceToken && (
                      <div className="absolute right-2 top-1/2 -translate-y-1/2">
                        {renderTokenStatus()}
                      </div>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={isInitializing || !isWorkerReady}
                  >
                    {isInitializing ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <Key className="h-4 w-4 mr-2" />
                    )}
                    {isInitializing ? "Validating..." : "Save & Validate"}
                  </Button>
                </div>
              </div>
              {!isWorkerReady && (
                <div className="flex items-center gap-2 text-sm text-amber-500 dark:text-amber-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Initializing worker...</span>
                </div>
              )}
            </div>
          </form>

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
                <div className="space-x-0.5 text-muted-foreground">
                  <span className="text-sm">
                    {longDate.format(APP_VERSION_DATE)}
                  </span>
                  {" · "}
                  <Badge variant="outline">v{APP_VERSION}</Badge>
                </div>
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

function useDarkThemePreference() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  useEffect(() => {
    const storedTheme = localStorage.getItem(THEME_KEY);
    const isDark =
      storedTheme === "dark" ||
      (!storedTheme &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);

    setIsDarkMode(isDark);

    // Make sure the UI matches the actual state
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleDarkMode = () => {
    const newDarkModeState = !isDarkMode;
    setIsDarkMode(newDarkModeState);

    if (newDarkModeState) {
      document.documentElement.classList.add("dark");
      localStorage.setItem(THEME_KEY, "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem(THEME_KEY, "light");
    }
  };

  return {
    isDarkMode,
    toggleDarkMode,
  };
}

export default Settings;
