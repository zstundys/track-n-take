import { RxJsonSchema } from "rxdb";

export interface PantryItem {
  id: string;
  name: string;
  quantity: number;
  unit: string;
  categoryId: string;
  expirationDate: number | null; // timestamp
  purchaseDate: number | null; // timestamp
  notes: string;
  isFinished: boolean;
  image?: string;
  barcode?: string;
  createdAt: number; // timestamp
  updatedAt: number; // timestamp
}

export const pantryItemSchema: RxJsonSchema<PantryItem> = {
  version: 0,
  primaryKey: "id",
  type: "object",
  properties: {
    id: {
      type: "string",
      maxLength: 100,
    },
    name: {
      type: "string",
    },
    quantity: {
      type: "number",
      minimum: 0,
    },
    unit: {
      type: "string",
      default: "item",
    },
    categoryId: {
      type: "string",
    },
    expirationDate: {
      type: ["number", "null"],
      default: null,
    },
    purchaseDate: {
      type: ["number", "null"],
      default: null,
    },
    notes: {
      type: "string",
      default: "",
    },
    isFinished: {
      type: "boolean",
      default: false,
    },
    image: {
      type: ["string", "null"],
      default: null,
    },
    barcode: {
      type: ["string", "null"],
      default: null,
    },
    createdAt: {
      type: "number",
    },
    updatedAt: {
      type: "number",
    },
  },
  required: ["id", "name", "quantity", "categoryId", "createdAt", "updatedAt"],
};

export type Color =
  | "green"
  | "blue"
  | "red"
  | "yellow"
  | "gray"
  | "orange"
  | "purple"
  | "cyan";

export interface Category {
  id: string;
  name: string;
  color: Color;
  icon: string;
  createdAt: number;
}

export interface ShoppingListItem {
  id: string;
  name: string;
  quantity: number;
  unit: string;
  categoryId: string;
  isChecked: boolean;
  fromPantryItemId: string | null;
  createdAt: number;
}

export type SortOption =
  | "name"
  | "expirationDate"
  | "purchaseDate"
  | "category";
export type FilterOption = "all" | "expiring" | "expired" | "finished";
export type Unit = "item" | "pack" | "kg" | "liter" | "other";

export const translateUnit = (
  t: (key: string) => string,
  unit: string
): string => {
  const unitToTranslationKeyMap: Record<Unit, string> = {
    item: "shoppingList.addDialog.unitItem",
    kg: "shoppingList.addDialog.unitKg",
    liter: "shoppingList.addDialog.unitLiter",
    other: "shoppingList.addDialog.unitOther",
    pack: "shoppingList.addDialog.unitPack",
  };
  if (unit in unitToTranslationKeyMap) {
    return t(unitToTranslationKeyMap[unit]);
  }

  return unit;
};
