import { TFunction } from "i18next";
import { RxJsonSchema } from "rxdb";

export interface PantryItem {
  id: string;
  name: string;
  quantity: number;
  unit?: string;
  categoryId?: string | null;
  expirationDate?: number | null;
  notes?: string;
  isFinished?: boolean;
  image?: string | null;
  barcode?: string | null;
  createdAt: number; // timestamp
  updatedAt: number; // timestamp
}

export type Color =
  | "green"
  | "blue"
  | "red"
  | "yellow"
  | "gray"
  | "orange"
  | "purple"
  | "cyan";

export type CategoryId =
  | "fruits-vegetables"
  | "dairy-and-eggs"
  | "meat-fish"
  | "grains"
  | "canned-goods"
  | "spices"
  | "snacks"
  | "beverages"
  | "other";
export interface Category {
  id: string;
  name: string;
  color: Color;
  icon: string;
  createdAt: number;
}

export type SortOption = "name" | "expirationDate" | "category";
export type FilterOption = "all" | "expiring" | "expired" | "finished";
export type Unit = "item" | "pack" | "kg" | "liter" | "other";

export const translateUnit = (
  t: TFunction,
  unit: string | undefined,
  count: number
): string => {
  const unitToTranslationKeyMap: Record<Unit, string> = {
    item: "shoppingList.addDialog.unitItem",
    kg: "shoppingList.addDialog.unitKg",
    liter: "shoppingList.addDialog.unitLiter",
    other: "shoppingList.addDialog.unitOther",
    pack: "shoppingList.addDialog.unitPack",
  };

  if (!unit) {
    return "";
  }

  if (unit in unitToTranslationKeyMap) {
    return t(unitToTranslationKeyMap[unit as Unit], { count });
  }

  return unit;
};
