import { createRxDatabase, addRxPlugin } from "rxdb";
import { getRxStorageDexie } from "rxdb/plugins/storage-dexie";
import { RxDBQueryBuilderPlugin } from "rxdb/plugins/query-builder";
import { RxDBMigrationPlugin } from "rxdb/plugins/migration";
import { RxDBUpdatePlugin } from "rxdb/plugins/update";
import { pantryItemSchema } from "../types";

// Add plugins
addRxPlugin(RxDBQueryBuilderPlugin);
addRxPlugin(RxDBMigrationPlugin);
addRxPlugin(RxDBUpdatePlugin);

// Database schema
const categorySchema = {
  title: 'Category Schema',
  version: 0,
  type: 'object',
  primaryKey: 'id',
  properties: {
    id: { type: 'string' },
    name: { type: 'string' },
    color: { type: 'string' },
    icon: { type: 'string' },
    createdAt: { type: 'number' },
  },
  required: ['id', 'name', 'color', 'icon', 'createdAt'],
};

const shoppingItemSchema = {
  title: 'Shopping Item Schema',
  version: 0,
  type: 'object',
  primaryKey: 'id',
  properties: {
    id: { type: 'string' },
    name: { type: 'string' },
    quantity: { type: 'number' },
    unit: { type: 'string' },
    categoryId: { type: 'string' },
    isChecked: { type: 'boolean' },
    fromPantryItemId: { type: ['string', 'null'] },
    createdAt: { type: 'number' },
  },
  required: ['id', 'name', 'quantity', 'unit', 'categoryId', 'isChecked', 'createdAt'],
};

// Database instance
let dbPromise: Promise<any> | null = null;

export const getDatabase = async () => {
  if (dbPromise) return dbPromise;

  dbPromise = createRxDatabase({
    name: "pantrydb.v2",
    storage: getRxStorageDexie(),
  }).then(async (db) => {
    // Create collections
    await db.addCollections({
      items: {
        schema: pantryItemSchema,
        migrationStrategies: {
          // Add migrations as needed
          1: (oldDoc: any) => oldDoc, // Initial version
        },
      },
      categories: {
        schema: categorySchema,
      },
      shoppingList: {
        schema: shoppingItemSchema,
      },
    });

    // Initialize with default data if needed
    await initializeDefaultData(db);

    return db;
  });

  return dbPromise;
};

// Initialize default data
const initializeDefaultData = async (db: any) => {
  try {
    // Initialize with default categories if needed
    const categoriesCount = await db.categories.count().exec();

    if (categoriesCount === 0) {
      const now = Date.now();
      const defaultCategories = [
        {
          id: "fruits-vegetables",
          name: "Fruits & Vegetables", // Will be translated when displayed
          color: "green",
          icon: "apple",
          createdAt: now,
        },
        {
          id: "dairy",
          name: "Dairy", // Will be translated when displayed
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
      ];

      await db.categories.bulkInsert(defaultCategories);
    }
  } catch (error) {
    console.error('Error initializing default data:', error);
  }
};

export const syncLocalChanges = async () => {
  // This would connect to a remote server for sync
  // For now, we're just using local storage
  console.log("Local changes synced");
  return true;
};

export default getDatabase;
