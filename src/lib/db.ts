import {
  createRxDatabase,
  addRxPlugin,
  RxDatabase,
  RxCollection,
  RxJsonSchema,
  ExtractDocumentTypeFromTypedRxJsonSchema,
  toTypedRxJsonSchema,
} from "rxdb";
import { getRxStorageDexie } from "rxdb/plugins/storage-dexie";
import { RxDBQueryBuilderPlugin } from "rxdb/plugins/query-builder";
import { RxDBMigrationPlugin } from "rxdb/plugins/migration";
import { RxDBDevModePlugin } from "rxdb/plugins/dev-mode";
import { RxDBUpdatePlugin } from "rxdb/plugins/update";
import { PantryItem } from "@/types";

const pantryItemSchema = {
  version: 1,
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
      type: ["string", "null"],
      default: null,
    },
    expirationDate: {
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
  required: ["id", "name", "quantity", "createdAt", "updatedAt"],
} as const satisfies RxJsonSchema<PantryItem>;

type PantryItemDoc = ExtractDocumentTypeFromTypedRxJsonSchema<
  typeof pantryItemSchema
>;

if (process.env.NODE_ENV === "development") {
  addRxPlugin(RxDBDevModePlugin);
}
addRxPlugin(RxDBQueryBuilderPlugin);
addRxPlugin(RxDBMigrationPlugin);
addRxPlugin(RxDBUpdatePlugin);

// Database schema
const categorySchema = {
  title: "Category Schema",
  version: 0,
  type: "object",
  primaryKey: "id",
  properties: {
    id: { type: "string", maxLength: 100 },
    name: { type: "string" },
    color: { type: "string" },
    icon: { type: "string" },
    createdAt: { type: "number" },
  },
  required: ["id", "name", "color", "icon", "createdAt"],
} as const;

type CategoryDoc = ExtractDocumentTypeFromTypedRxJsonSchema<
  typeof categorySchema
>;

const shoppingItemSchema = {
  title: "Shopping Item Schema",
  version: 3,
  type: "object",
  primaryKey: "id",
  properties: {
    id: { type: "string", maxLength: 100 },
    name: { type: "string" },
    quantity: { type: "number" },
    unit: { type: "string" },
    categoryId: { type: "string" },
    isChecked: { type: "boolean" },
    fromPantryItemId: { type: ["string", "null"] },
    createdAt: { type: "number" },
  },
  required: [
    "id",
    "name",
    "quantity",
    "unit",
    "categoryId",
    "isChecked",
    "createdAt",
  ],
} as const;
type ShoppingItemDoc = ExtractDocumentTypeFromTypedRxJsonSchema<
  typeof shoppingItemSchema
>;

type AppRxDatabase = RxDatabase<{
  items: RxCollection<PantryItemDoc>;
  categories: RxCollection<CategoryDoc>;
  shopping_list: RxCollection<ShoppingItemDoc>;
}>;

// Database instance
let dbPromise: Promise<AppRxDatabase> | null = null;

export const getDatabase = async () => {
  if (dbPromise) return dbPromise;

  dbPromise = createRxDatabase<AppRxDatabase>({
    name: "pantrydb-v4",
    storage: getRxStorageDexie(),
  }).then(async (db) => {
    // Create collections
    await db.addCollections({
      items: {
        schema: pantryItemSchema,
        migrationStrategies: {
          1: (oldDoc) => oldDoc,
        },
      },
      categories: {
        schema: categorySchema,
      },
      shopping_list: {
        schema: shoppingItemSchema,
        migrationStrategies: {
          1: (oldDoc) => oldDoc,
          2: (oldDoc) => oldDoc,
          3: (oldDoc) => oldDoc,
        },
      },
    });

    // Initialize with default data if needed
    await initializeDefaultData(db);

    return db;
  });

  return dbPromise;
};

// Initialize default data
const initializeDefaultData = async (db: AppRxDatabase) => {
  try {
    // Initialize with default categories if needed
    const categoriesCount = await db.categories.count().exec();

    if (categoriesCount === 0) {
      const now = Date.now();

      await db.categories.bulkInsert([
        {
          id: "fruits-vegetables",
          name: "Fruits & Vegetables", // Will be translated when displayed
          color: "green",
          icon: "apple",
          createdAt: now,
        },
        {
          id: "dairy-and-eggs",
          name: "Dairy & Eggs", // Will be translated when displayed
          color: "blue",
          icon: "milk",
          createdAt: now + 1,
        },
        {
          id: "meat-fish",
          name: "Meat & Fish", // Will be translated when displayed
          color: "red",
          icon: "beef",
          createdAt: now + 2,
        },
        {
          id: "grains",
          name: "Grains & Pasta", // Will be translated when displayed
          color: "yellow",
          icon: "wheat",
          createdAt: now + 3,
        },
        {
          id: "canned-goods",
          name: "Canned Goods", // Will be translated when displayed
          color: "gray",
          icon: "can",
          createdAt: now + 4,
        },
        {
          id: "spices",
          name: "Spices & Herbs", // Will be translated when displayed
          color: "orange",
          icon: "spice",
          createdAt: now + 5,
        },
        {
          id: "snacks",
          name: "Snacks", // Will be translated when displayed
          color: "purple",
          icon: "cookie",
          createdAt: now + 6,
        },
        {
          id: "beverages",
          name: "Beverages", // Will be translated when displayed
          color: "cyan",
          icon: "bottle",
          createdAt: now + 7,
        },
        {
          id: "other",
          name: "Other", // Will be translated when displayed
          color: "gray",
          icon: "box",
          createdAt: now + 8,
        },
      ]);
    }
  } catch (error) {
    console.error("Error initializing default data:", error);
  }
};

export const syncLocalChanges = async () => {
  // This would connect to a remote server for sync
  // For now, we're just using local storage
  console.log("Local changes synced");
  return true;
};

export default getDatabase;
