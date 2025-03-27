import { useMemo } from "react";
import { useTranslation } from "react-i18next";

type DateTimeFormatOptions = Intl.DateTimeFormatOptions;

export function useIntl() {
  const { i18n } = useTranslation();

  const instanceLongDate = useMemo(
    () =>
      new Intl.DateTimeFormat(i18n.language, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }),
    [i18n.language]
  );

  return {
    /** @example "Jan 1, 2023, 12:00 AM" */
    longDate: instanceLongDate,
  };
}
