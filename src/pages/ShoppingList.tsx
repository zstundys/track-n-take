
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Checkbox } from '@/components/ui/checkbox';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui-custom/Badge';
import AppLayout from '@/components/layout/AppLayout';
import { useShoppingList } from '@/hooks/useShoppingList';
import { ShoppingBag, Plus, Trash2, Search, CheckSquare, XCircle } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { 
  Dialog, 
  DialogContent, 
  DialogFooter, 
  DialogHeader, 
  DialogTitle 
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

const ShoppingList: React.FC = () => {
  const {
    items,
    categories,
    isLoading,
    addItem,
    toggleItemCheck,
    deleteItem,
    clearCheckedItems,
    addFromPantry,
  } = useShoppingList();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  
  // New item form state
  const [newItemName, setNewItemName] = useState('');
  const [newItemQuantity, setNewItemQuantity] = useState(1);
  const [newItemUnit, setNewItemUnit] = useState('item');
  const [newItemCategory, setNewItemCategory] = useState('');
  
  // Filter items by search query
  const filteredItems = items.filter((item) =>
    item.name.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Get category for an item
  const getCategoryForItem = (categoryId: string) => {
    return (
      categories.find((cat) => cat.id === categoryId) || {
        id: 'unknown',
        name: 'Unknown',
        color: 'gray',
        icon: 'box',
        createdAt: 0,
      }
    );
  };
  
  // Handle add new item
  const handleAddItem = async () => {
    if (!newItemName.trim()) return;
    
    await addItem({
      name: newItemName,
      quantity: newItemQuantity,
      unit: newItemUnit,
      categoryId: newItemCategory || 'other',
      isChecked: false,
      fromPantryItemId: null,
    });
    
    // Reset form
    setNewItemName('');
    setNewItemQuantity(1);
    setNewItemUnit('item');
    setNewItemCategory('');
    setIsAddDialogOpen(false);
  };
  
  // Calculate stats
  const totalItems = items.length;
  const checkedItems = items.filter(item => item.isChecked).length;
  
  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
          <div>
            <h1 className="text-3xl font-medium">Shopping List</h1>
            <p className="text-muted-foreground">
              {checkedItems} of {totalItems} items checked
            </p>
          </div>
          
          <div className="flex gap-2 self-end">
            <Button 
              variant="outline" 
              size="sm" 
              className="gap-1"
              onClick={() => addFromPantry('all')}
            >
              <ShoppingBag className="h-4 w-4" />
              <span>Add from Pantry</span>
            </Button>
            
            <Button 
              variant="default" 
              size="sm" 
              className="gap-1"
              onClick={() => setIsAddDialogOpen(true)}
            >
              <Plus className="h-4 w-4" />
              <span>Add Item</span>
            </Button>
          </div>
        </div>
        
        <div className="flex items-center justify-between mb-6 gap-4">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search items..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-full rounded-full bg-background border-input"
            />
          </div>
          
          {checkedItems > 0 && (
            <Button 
              variant="outline" 
              size="sm" 
              className="gap-1 shrink-0"
              onClick={() => clearCheckedItems()}
            >
              <Trash2 className="h-4 w-4" />
              <span className="hidden sm:inline">Clear Checked</span>
            </Button>
          )}
        </div>
        
        <AnimatePresence>
          {filteredItems.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="text-center py-12"
            >
              <ShoppingBag className="h-12 w-12 mx-auto text-muted-foreground/50 mb-4" />
              <h3 className="text-xl font-medium mb-2">Your shopping list is empty</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery
                  ? "No items match your search"
                  : "Add items to your shopping list"}
              </p>
              <Button onClick={() => setIsAddDialogOpen(true)}>
                Add Your First Item
              </Button>
            </motion.div>
          ) : (
            <ul className="space-y-2">
              {filteredItems.map((item) => {
                const category = getCategoryForItem(item.categoryId);
                
                return (
                  <motion.li
                    key={item.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -100 }}
                    transition={{ duration: 0.2 }}
                    layout
                  >
                    <Card className={`p-3 flex items-center ${item.isChecked ? 'bg-secondary/50' : ''}`}>
                      <Checkbox
                        checked={item.isChecked}
                        onCheckedChange={(checked) => 
                          toggleItemCheck(item.id, checked === true)
                        }
                        className="mr-3"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant={category.color as any} size="sm">
                            {category.name}
                          </Badge>
                        </div>
                        <p className={`font-medium ${item.isChecked ? 'line-through text-muted-foreground' : ''}`}>
                          {item.name}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {item.quantity} {item.unit}
                        </p>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteItem(item.id)}
                        className="ml-2 text-muted-foreground hover:text-destructive"
                      >
                        <XCircle className="h-5 w-5" />
                      </Button>
                    </Card>
                  </motion.li>
                );
              })}
            </ul>
          )}
        </AnimatePresence>
      </motion.div>
      
      {/* Add Item Dialog */}
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add New Item</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <label htmlFor="name" className="text-sm font-medium">
                Item Name
              </label>
              <Input
                id="name"
                value={newItemName}
                onChange={(e) => setNewItemName(e.target.value)}
                placeholder="Enter item name"
                autoComplete="off"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="grid gap-2">
                <label htmlFor="quantity" className="text-sm font-medium">
                  Quantity
                </label>
                <Input
                  id="quantity"
                  type="number"
                  min="1"
                  value={newItemQuantity}
                  onChange={(e) => setNewItemQuantity(Number(e.target.value))}
                />
              </div>
              <div className="grid gap-2">
                <label htmlFor="unit" className="text-sm font-medium">
                  Unit
                </label>
                <Input
                  id="unit"
                  value={newItemUnit}
                  onChange={(e) => setNewItemUnit(e.target.value)}
                  placeholder="e.g., pack, bottle, lb"
                />
              </div>
            </div>
            
            <div className="grid gap-2">
              <label htmlFor="category" className="text-sm font-medium">
                Category
              </label>
              <Select
                value={newItemCategory}
                onValueChange={setNewItemCategory}
              >
                <SelectTrigger id="category">
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent>
                  {categories.map((category) => (
                    <SelectItem key={category.id} value={category.id}>
                      {category.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddItem}>Add Item</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  );
};

export default ShoppingList;
