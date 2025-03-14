
import { useState, useEffect } from 'react';
import { useToast } from '@/components/ui/use-toast';
import getDatabase from '@/lib/db';
import { PantryItem, Category, FilterOption, SortOption } from '@/types';
import { v4 as uuidv4 } from 'uuid';

export const usePantryItems = (initialFilter: FilterOption = 'all', initialSort: SortOption = 'name') => {
  const [items, setItems] = useState<PantryItem[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<FilterOption>(initialFilter);
  const [sortBy, setSortBy] = useState<SortOption>(initialSort);
  const { toast } = useToast();
  
  // Load items and categories
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        const db = await getDatabase();
        
        // Subscribe to items
        const itemsSub = db.items.find().$.subscribe((foundItems: PantryItem[]) => {
          // Apply filtering
          let filteredItems = [...foundItems];
          
          if (filter === 'expired') {
            filteredItems = filteredItems.filter(item => 
              item.expirationDate && item.expirationDate < Date.now()
            );
          } else if (filter === 'expiring') {
            const sevenDaysFromNow = Date.now() + 7 * 24 * 60 * 60 * 1000;
            filteredItems = filteredItems.filter(item => 
              item.expirationDate && 
              item.expirationDate > Date.now() && 
              item.expirationDate < sevenDaysFromNow
            );
          } else if (filter === 'finished') {
            filteredItems = filteredItems.filter(item => item.isFinished);
          }
          
          // Apply sorting
          filteredItems.sort((a, b) => {
            if (sortBy === 'name') {
              return a.name.localeCompare(b.name);
            } else if (sortBy === 'expirationDate') {
              const aDate = a.expirationDate || Number.MAX_SAFE_INTEGER;
              const bDate = b.expirationDate || Number.MAX_SAFE_INTEGER;
              return aDate - bDate;
            } else if (sortBy === 'purchaseDate') {
              const aDate = a.purchaseDate || 0;
              const bDate = b.purchaseDate || 0;
              return bDate - aDate; // Newest first
            } else if (sortBy === 'category') {
              return a.categoryId.localeCompare(b.categoryId);
            }
            return 0;
          });
          
          setItems(filteredItems);
          setIsLoading(false);
        });
        
        // Subscribe to categories
        const categoriesSub = db.categories.find().$.subscribe((foundCategories: Category[]) => {
          setCategories(foundCategories.sort((a, b) => a.name.localeCompare(b.name)));
        });
        
        return () => {
          itemsSub.unsubscribe();
          categoriesSub.unsubscribe();
        };
      } catch (error) {
        console.error('Error loading data:', error);
        toast({
          title: 'Error',
          description: 'Failed to load pantry items. Please try again.',
          variant: 'destructive',
        });
        setIsLoading(false);
      }
    };
    
    loadData();
  }, [filter, sortBy, toast]);
  
  // Add item
  const addItem = async (itemData: Omit<PantryItem, 'id' | 'createdAt' | 'updatedAt'>) => {
    try {
      const db = await getDatabase();
      const now = Date.now();
      
      const newItem: PantryItem = {
        ...itemData,
        id: uuidv4(),
        createdAt: now,
        updatedAt: now,
      };
      
      await db.items.insert(newItem);
      
      toast({
        title: 'Item Added',
        description: `${newItem.name} has been added to your pantry.`,
      });
      
      return newItem;
    } catch (error) {
      console.error('Error adding item:', error);
      toast({
        title: 'Error',
        description: 'Failed to add item. Please try again.',
        variant: 'destructive',
      });
      return null;
    }
  };
  
  // Update item
  const updateItem = async (id: string, updates: Partial<PantryItem>) => {
    try {
      const db = await getDatabase();
      const item = await db.items.findOne(id).exec();
      
      if (!item) {
        throw new Error('Item not found');
      }
      
      await item.update({
        $set: {
          ...updates,
          updatedAt: Date.now(),
        }
      });
      
      toast({
        title: 'Item Updated',
        description: `${item.name} has been updated.`,
      });
      
      return true;
    } catch (error) {
      console.error('Error updating item:', error);
      toast({
        title: 'Error',
        description: 'Failed to update item. Please try again.',
        variant: 'destructive',
      });
      return false;
    }
  };
  
  // Delete item
  const deleteItem = async (id: string) => {
    try {
      const db = await getDatabase();
      const item = await db.items.findOne(id).exec();
      
      if (!item) {
        throw new Error('Item not found');
      }
      
      await item.remove();
      
      toast({
        title: 'Item Removed',
        description: `${item.name} has been removed from your pantry.`,
      });
      
      return true;
    } catch (error) {
      console.error('Error deleting item:', error);
      toast({
        title: 'Error',
        description: 'Failed to delete item. Please try again.',
        variant: 'destructive',
      });
      return false;
    }
  };
  
  // Add to shopping list
  const addToShoppingList = async (item: PantryItem) => {
    try {
      const db = await getDatabase();
      const now = Date.now();
      
      await db.shoppingList.insert({
        id: uuidv4(),
        name: item.name,
        quantity: 1,
        unit: item.unit,
        categoryId: item.categoryId,
        isChecked: false,
        fromPantryItemId: item.id,
        createdAt: now,
      });
      
      toast({
        title: 'Added to Shopping List',
        description: `${item.name} has been added to your shopping list.`,
      });
      
      return true;
    } catch (error) {
      console.error('Error adding to shopping list:', error);
      toast({
        title: 'Error',
        description: 'Failed to add item to shopping list. Please try again.',
        variant: 'destructive',
      });
      return false;
    }
  };
  
  return {
    items,
    categories,
    isLoading,
    filter,
    setFilter,
    sortBy,
    setSortBy,
    addItem,
    updateItem,
    deleteItem,
    addToShoppingList,
  };
};
