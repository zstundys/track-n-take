import { Color } from "@/types";
import { useTranslation } from "react-i18next";

/**
 * Maps database category IDs to translation keys
 */
const categoryTranslationMap: Record<string, string> = {
  "fruits-vegetables": "categories.fruitsVegetables",
  dairy: "categories.dairy",
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
  const cLight = (
    colorCode: string
  ): { background: string; text: string; border: string } => {
    return {
      background: `bg-${colorCode}-200`,
      text: `text-${colorCode}-800`,
      border: `border-${colorCode}-300`,
    };
  };
  const cDark = (
    colorCode: string
  ): { background: string; text: string; border: string } => {
    return {
      background: `bg-${colorCode}-800`,
      text: `dark:text-${colorCode}-200`,
      border: `dark:border-${colorCode}-600`,
    };
  };

  const colorMapping = {
    light: {
      gray: cLight("gray"),
      blue: cLight("blue"),
      cyan: cLight("cyan"),
      green: cLight("green"),
      orange: cLight("orange"),
      purple: cLight("purple"),
      red: cLight("red"),
      yellow: cLight("yellow"),
    } satisfies Record<
      Color,
      { background: string; text: string; border: string }
    >,
    dark: {
      gray: cDark("slate"),
      blue: cDark("blue"),
      cyan: cDark("cyan"),
      green: cDark("green"),
      orange: cDark("orange"),
      purple: cDark("purple"),
      red: cDark("red"),
      yellow: cDark("yellow"),
    } satisfies Record<
      Color,
      { background: string; text: string; border: string }
    >,
  };

  return `${Object.values(colorMapping.light[color]).join(" ")} ${Object.values(
    colorMapping.dark[color]
  ).join(" ")}`;
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
