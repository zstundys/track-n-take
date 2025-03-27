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
