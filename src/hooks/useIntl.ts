import { useMemo } from "react";
import { useTranslation } from "react-i18next";

type IntlAPI = {
  /** @example "Jan 1, 2023, 12:00 AM" */
  longDate: Intl.DateTimeFormat;

  /** @example "Jan 1, 2023" */
  shortDate: Intl.DateTimeFormat;
};

export function useIntl(): IntlAPI {
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

  const instanceShortDate = useMemo(
    () =>
      new Intl.DateTimeFormat(i18n.language, {
        year: "numeric",
        month: "short",
        day: "numeric",
      }),
    [i18n.language]
  );

  return {
    /** @example "Jan 1, 2023, 12:00 AM" */
    longDate: instanceLongDate,
    shortDate: instanceShortDate,
  };
}
