import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import LanguageDetector from "i18next-browser-languagedetector";
import ICU from "i18next-icu";

// Import translations
import enTranslation from "./locales/en.json";
import ltTranslation from "./locales/lt.json";

// Configure i18next
i18n
  // Detect user language
  .use(LanguageDetector)
  // Add ICU format support - must be before initReactI18next
  .use(new ICU())
  // Pass the i18n instance to react-i18next
  .use(initReactI18next)
  // Initialize i18next
  .init({
    resources: {
      en: {
        translation: enTranslation,
      },
      lt: {
        translation: ltTranslation,
      },
    },
    fallbackLng: "en",
    debug: true,

    // Common namespace used around the app
    ns: ["translation"],
    defaultNS: "translation",

    interpolation: {
      escapeValue: false, // React already safes from XSS
    },

    // Language detection options
    detection: {
      order: ["localStorage", "navigator"],
      lookupLocalStorage: "i18nextLng",
      caches: ["localStorage"],
    },
  });

export default i18n;
