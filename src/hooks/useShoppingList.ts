import { useState, useEffect } from "react";
import { useToast } from "@/components/ui/use-toast";
import getDatabase from "@/lib/db";
import { ShoppingListItem, Category } from "@/types";
import { v4 as uuidv4 } from "uuid";

export const useShoppingList = () => {
  const [items, setItems] = useState<ShoppingListItem[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  // Load items and categories
  useEffect(() => {
    const loadData = async () => {
      try {
        setIsLoading(true);
        const db = await getDatabase();

        // Subscribe to shopping list items
        const itemsSub = db.shopping_list.find().$.subscribe((foundItems) => {
          // Sort items: unchecked first, then by category, then by name
          const sortedItems = [...foundItems].sort((a, b) => {
            // First sort by checked status
            if (a.isChecked !== b.isChecked) {
              return a.isChecked ? 1 : -1;
            }

            // Then by category
            if (a.categoryId !== b.categoryId) {
              return a.categoryId.localeCompare(b.categoryId);
            }

            // Finally by name
            return a.name.localeCompare(b.name);
          });

          setItems(sortedItems);
          setIsLoading(false);
        });

        // Subscribe to categories
        const categoriesSub = db.categories
          .find()
          .$.subscribe((foundCategories: Category[]) => {
            setCategories(
              foundCategories.sort((a, b) => a.name.localeCompare(b.name))
            );
          });

        return () => {
          itemsSub.unsubscribe();
          categoriesSub.unsubscribe();
        };
      } catch (error) {
        console.error("Error loading shopping list:", error);
        toast({
          title: "Error",
          description: "Failed to load shopping list. Please try again.",
          variant: "destructive",
        });
        setIsLoading(false);
      }
    };

    loadData();
  }, [toast]);

  // Add item
  const addItem = async (
    itemData: Omit<ShoppingListItem, "id" | "createdAt">
  ) => {
    try {
      const db = await getDatabase();
      const now = Date.now();

      const newItem: ShoppingListItem = {
        ...itemData,
        id: uuidv4(),
        createdAt: now,
      };

      await db.shopping_list.insert(newItem);

      toast({
        title: "Item Added",
        description: `${newItem.name} has been added to your shopping list.`,
      });

      return newItem;
    } catch (error) {
      console.error("Error adding item to shopping list:", error);
      toast({
        title: "Error",
        description: "Failed to add item. Please try again.",
        variant: "destructive",
      });
      return null;
    }
  };

  // Toggle item check
  const toggleItemCheck = async (id: string, isChecked: boolean) => {
    try {
      const db = await getDatabase();
      const item = await db.shopping_list.findOne(id).exec();

      if (!item) {
        throw new Error("Item not found");
      }

      await item.update({
        $set: {
          isChecked,
        },
      });

      return true;
    } catch (error) {
      console.error("Error updating shopping list item:", error);
      toast({
        title: "Error",
        description: "Failed to update item. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  // Update item
  const updateItem = async (id: string, updates: Partial<ShoppingListItem>) => {
    try {
      const db = await getDatabase();
      const item = await db.shopping_list.findOne(id).exec();

      if (!item) {
        throw new Error("Item not found");
      }

      await item.update({
        $set: updates,
      });

      toast({
        title: "Item Updated",
        description: `${item.name} has been updated.`,
      });

      return true;
    } catch (error) {
      console.error("Error updating shopping list item:", error);
      toast({
        title: "Error",
        description: "Failed to update item. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  // Delete item
  const deleteItem = async (id: string) => {
    try {
      const db = await getDatabase();
      const item = await db.shopping_list.findOne(id).exec();

      if (!item) {
        throw new Error("Item not found");
      }

      await item.remove();

      toast({
        title: "Item Removed",
        description: `${item.name} has been removed from your shopping list.`,
      });

      return true;
    } catch (error) {
      console.error("Error deleting shopping list item:", error);
      toast({
        title: "Error",
        description: "Failed to delete item. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  // Clear checked items
  const clearCheckedItems = async () => {
    try {
      const db = await getDatabase();
      const checkedItems = await db.shopping_list
        .find({
          selector: {
            isChecked: true,
          },
        })
        .exec();

      if (checkedItems.length === 0) {
        return true;
      }

      await Promise.all(checkedItems.map((item) => item.remove()));

      toast({
        title: "List Cleared",
        description: `${checkedItems.length} checked items have been removed.`,
      });

      return true;
    } catch (error) {
      console.error("Error clearing checked items:", error);
      toast({
        title: "Error",
        description: "Failed to clear checked items. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  // Add pantry items to shopping list
  const addFromPantry = async (
    filter: "finished" | "expired" | "all" = "all"
  ) => {
    try {
      const db = await getDatabase();

      // Get pantry items that match the filter
      const selector: any = {};

      if (filter === "finished") {
        selector.isFinished = true;
      } else if (filter === "expired") {
        selector.expirationDate = {
          $lt: Date.now(),
        };
      }

      const pantryItems = await db.items
        .find({
          selector,
        })
        .exec();

      if (pantryItems.length === 0) {
        toast({
          title: "No Items Found",
          description: "No matching items to add to your shopping list.",
        });
        return false;
      }

      // Create shopping list items
      const now = Date.now();
      const shoppingItems = pantryItems.map((item) => ({
        id: uuidv4(),
        name: item.name,
        quantity: 1,
        unit: item.unit,
        categoryId: item.categoryId,
        isChecked: false,
        fromPantryItemId: item.id,
        createdAt: now,
      }));

      await db.shopping_list.bulkInsert(shoppingItems);

      toast({
        title: "Items Added",
        description: `${shoppingItems.length} items added to your shopping list.`,
      });

      return true;
    } catch (error) {
      console.error("Error adding pantry items to shopping list:", error);
      toast({
        title: "Error",
        description: "Failed to add items. Please try again.",
        variant: "destructive",
      });
      return false;
    }
  };

  return {
    items,
    categories,
    isLoading,
    addItem,
    toggleItemCheck,
    updateItem,
    deleteItem,
    clearCheckedItems,
    addFromPantry,
  };
};
