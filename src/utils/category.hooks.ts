import { CategoryId, Color } from "@/types";
import { useTranslation } from "react-i18next";

/**
 * Maps database category IDs to translation keys
 */
const categoryTranslationMap: Record<CategoryId, string> = {
  "fruits-vegetables": "categories.fruitsVegetables",
  "dairy-and-eggs": "categories.dairy",
  "meat-fish": "categories.meatFish",
  grains: "categories.grains",
  "canned-goods": "categories.cannedGoods",
  spices: "categories.spicesHerbs",
  snacks: "categories.snacks",
  beverages: "categories.beverages",
  other: "categories.other",
};

/**
 * Hook to translate category names
 * @param categoryId The ID of the category to translate
 * @returns The translated category name
 */
export const useTranslateCategory = (categoryId: string): string => {
  const { t } = useTranslation();

  // If the category ID is in our map, translate it
  if (categoryId in categoryTranslationMap) {
    return t(categoryTranslationMap[categoryId]);
  }

  // If not found, return the ID as a fallback
  return categoryId;
};

export const getCategoryColorClasses = (color: Color) => {
  const colorMapping = {
    light: {
      gray: {
        background: `bg-gray-200`,
        text: `text-gray-800`,
        border: `border-gray-300`,
      },
      blue: {
        background: `bg-blue-200`,
        text: `text-blue-800`,
        border: `border-blue-300`,
      },
      cyan: {
        background: `bg-cyan-200`,
        text: `text-cyan-800`,
        border: `border-cyan-300`,
      },
      green: {
        background: `bg-green-200`,
        text: `text-green-800`,
        border: `border-green-300`,
      },
      orange: {
        background: `bg-orange-200`,
        text: `text-orange-800`,
        border: `border-orange-300`,
      },
      purple: {
        background: `bg-purple-200`,
        text: `text-purple-800`,
        border: `border-purple-300`,
      },
      red: {
        background: `bg-red-200`,
        text: `text-red-800`,
        border: `border-red-300`,
      },
      yellow: {
        background: `bg-yellow-200`,
        text: `text-yellow-800`,
        border: `border-yellow-300`,
      },
    } satisfies Record<
      Color,
      { background: string; text: string; border: string }
    >,
    dark: {
      gray: {
        background: `dark:bg-slate-800`,
        text: `dark:text-slate-200`,
        border: `dark:border-slate-600`,
      },
      blue: {
        background: `dark:bg-blue-800`,
        text: `dark:text-blue-200`,
        border: `dark:border-blue-600`,
      },
      cyan: {
        background: `dark:bg-cyan-800`,
        text: `dark:text-cyan-200`,
        border: `dark:border-cyan-600`,
      },
      green: {
        background: `dark:bg-green-800`,
        text: `dark:text-green-200`,
        border: `dark:border-green-600`,
      },
      orange: {
        background: `dark:bg-orange-800`,
        text: `dark:text-orange-200`,
        border: `dark:border-orange-600`,
      },
      purple: {
        background: `dark:bg-purple-800`,
        text: `dark:text-purple-200`,
        border: `dark:border-purple-600`,
      },
      red: {
        background: `dark:bg-red-800`,
        text: `dark:text-red-200`,
        border: `dark:border-red-600`,
      },
      yellow: {
        background: `dark:bg-yellow-800`,
        text: `dark:text-yellow-200`,
        border: `dark:border-yellow-600`,
      },
    } satisfies Record<
      Color,
      { background: string; text: string; border: string }
    >,
  };

  const classes = `${Object.values(colorMapping.light[color]).join(
    " "
  )} ${Object.values(colorMapping.dark[color]).join(" ")}`;

  return classes;
};

/**
 * Function to translate a category name (for use outside of React components)
 * @param t Translation function from useTranslation
 * @param categoryId The ID of the category to translate
 * @returns The translated category name
 */
export const translateCategory = (
  t: (key: string) => string,
  categoryId: string
): string => {
  // If the category ID is in our map, translate it
  if (categoryId in categoryTranslationMap) {
    return t(categoryTranslationMap[categoryId]);
  }

  // If not found, return the ID as a fallback
  return categoryId;
};
