
import { createRxDatabase, addRxPlugin } from 'rxdb';
import { getRxStorageMemory } from 'rxdb/plugins/storage-memory';
import { RxDBQueryBuilderPlugin } from 'rxdb/plugins/query-builder';
import { RxDBMigrationPlugin } from 'rxdb/plugins/migration';
import { pantryItemSchema } from '../types';

// Add plugins
addRxPlugin(RxDBQueryBuilderPlugin);
addRxPlugin(RxDBMigrationPlugin);

// Database instance
let dbPromise: Promise<any> | null = null;

export const getDatabase = async () => {
  if (dbPromise) return dbPromise;

  dbPromise = createRxDatabase({
    name: 'pantrydb',
    storage: getRxStorageMemory()
  }).then(async (db) => {
    // Create collections
    await db.addCollections({
      items: {
        schema: pantryItemSchema,
        migrationStrategies: {
          // Add migrations as needed
          1: (oldDoc: any) => oldDoc, // Initial version
        }
      },
      categories: {
        schema: {
          version: 0,
          primaryKey: 'id',
          type: 'object',
          properties: {
            id: {
              type: 'string',
              maxLength: 100
            },
            name: {
              type: 'string'
            },
            color: {
              type: 'string',
              default: 'gray'
            },
            icon: {
              type: 'string',
              default: 'box'
            },
            createdAt: {
              type: 'number'
            }
          },
          required: ['id', 'name', 'createdAt']
        }
      },
      shoppingList: {
        schema: {
          version: 0,
          primaryKey: 'id',
          type: 'object',
          properties: {
            id: {
              type: 'string',
              maxLength: 100
            },
            name: {
              type: 'string'
            },
            quantity: {
              type: 'number',
              minimum: 1
            },
            unit: {
              type: 'string',
              default: 'item'
            },
            categoryId: {
              type: 'string'
            },
            isChecked: {
              type: 'boolean',
              default: false
            },
            fromPantryItemId: {
              type: ['string', 'null'],
              default: null
            },
            createdAt: {
              type: 'number'
            }
          },
          required: ['id', 'name', 'quantity', 'createdAt']
        }
      }
    });

    // Initialize with default categories if needed
    const categoriesCount = await db.categories.count().exec();
    
    if (categoriesCount === 0) {
      const now = Date.now();
      const defaultCategories = [
        { id: 'fruits-vegetables', name: 'Fruits & Vegetables', color: 'green', icon: 'apple', createdAt: now },
        { id: 'dairy', name: 'Dairy', color: 'blue', icon: 'milk', createdAt: now + 1 },
        { id: 'meat-fish', name: 'Meat & Fish', color: 'red', icon: 'beef', createdAt: now + 2 },
        { id: 'grains', name: 'Grains & Pasta', color: 'yellow', icon: 'wheat', createdAt: now + 3 },
        { id: 'canned-goods', name: 'Canned Goods', color: 'gray', icon: 'can', createdAt: now + 4 },
        { id: 'spices', name: 'Spices & Herbs', color: 'orange', icon: 'spice', createdAt: now + 5 },
        { id: 'snacks', name: 'Snacks', color: 'purple', icon: 'cookie', createdAt: now + 6 },
        { id: 'beverages', name: 'Beverages', color: 'cyan', icon: 'bottle', createdAt: now + 7 },
        { id: 'other', name: 'Other', color: 'gray', icon: 'box', createdAt: now + 8 }
      ];
      
      await db.categories.bulkInsert(defaultCategories);
    }

    return db;
  });

  return dbPromise;
};

export const syncLocalChanges = async () => {
  // This would connect to a remote server for sync
  // For now, we're just using local storage
  console.log('Local changes synced');
  return true;
};

export default getDatabase;
