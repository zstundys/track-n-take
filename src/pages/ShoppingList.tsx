import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
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
import { cn } from '@/lib/utils';
import { translateCategory } from '@/utils/categoryTranslation';

const ShoppingList: React.FC = () => {
  const { t } = useTranslation();
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
  const [newItemUnit, setNewItemUnit] = useState(t('shoppingList.addDialog.unitItem'));
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
        name: t('common.unknown'),
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
    setNewItemUnit(t('shoppingList.addDialog.unitItem'));
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
            <h1 className="text-3xl font-medium">{t('shoppingList.title')}</h1>
            <p className="text-muted-foreground">
              {t('shoppingList.stats', { checked: checkedItems, total: totalItems })}
            </p>
          </div>
          
          <div className="flex gap-2 w-full sm:w-auto">
            <div className="relative flex-1 sm:flex-initial">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
              <Input
                placeholder={t('shoppingList.search')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 w-full sm:w-64 rounded-full bg-background border-input"
              />
            </div>
            <Button 
              size="sm" 
              className="gap-2"
              onClick={() => setIsAddDialogOpen(true)}
            >
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">{t('shoppingList.addItem')}</span>
            </Button>
            <Button 
              size="sm" 
              variant="outline" 
              className="gap-2"
              onClick={clearCheckedItems}
              disabled={checkedItems === 0}
            >
              <CheckSquare className="h-4 w-4" />
              <span className="hidden sm:inline">{t('shoppingList.clearChecked')}</span>
            </Button>
          </div>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center py-8">
            <ShoppingBag className="h-8 w-8 animate-pulse text-primary" />
            <span className="sr-only">{t('shoppingList.loading')}</span>
          </div>
        ) : filteredItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-primary/10 rounded-full p-6 mb-4">
              <ShoppingBag className="h-12 w-12 text-primary" />
            </div>
            <h2 className="text-2xl font-medium mb-2">{t('shoppingList.empty')}</h2>
            <p className="text-muted-foreground max-w-md mb-6">
              {t('shoppingList.emptyDescription')}
            </p>
            <Button 
              size="lg" 
              className="gap-2"
              onClick={() => setIsAddDialogOpen(true)}
            >
              <Plus className="h-5 w-5" />
              <span>{t('shoppingList.addFirst')}</span>
            </Button>
          </div>
        ) : (
          <div className="space-y-2">
            <AnimatePresence initial={false}>
              {filteredItems.map((item) => {
                const category = getCategoryForItem(item.categoryId);
                
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Card className={cn(
                      "flex items-center p-3 gap-3",
                      item.isChecked && "bg-muted/50"
                    )}>
                      <Checkbox 
                        checked={item.isChecked}
                        onCheckedChange={(checked) => toggleItemCheck(item.id,Boolean(checked))}
                        className="h-5 w-5"
                      />
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={cn(
                            "font-medium",
                            item.isChecked && "line-through text-muted-foreground"
                          )}>
                            {item.name}
                          </span>
                          <Badge variant={category.color as any} className="text-[10px]">
                            {translateCategory(t, category.id)}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {item.quantity} {item.unit}
                          {item.quantity > 1 && item.unit !== t('shoppingList.addDialog.unitKg') && item.unit !== t('shoppingList.addDialog.unitLiter') ? 's' : ''}
                        </div>
                      </div>
                      
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-muted-foreground hover:text-destructive"
                        onClick={() => deleteItem(item.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </Card>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </motion.div>
      
      <Dialog open={isAddDialogOpen} onOpenChange={setIsAddDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t('shoppingList.addDialog.title')}</DialogTitle>
          </DialogHeader>
          
          <div className="grid gap-4 py-4">
            <div className="space-y-2">
              <label htmlFor="item-name" className="text-sm font-medium">
                {t('shoppingList.addDialog.nameLabel')}
              </label>
              <Input
                id="item-name"
                value={newItemName}
                onChange={(e) => setNewItemName(e.target.value)}
                placeholder={t('shoppingList.addDialog.namePlaceholder')}
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label htmlFor="item-quantity" className="text-sm font-medium">
                  {t('shoppingList.addDialog.quantityLabel')}
                </label>
                <Input
                  id="item-quantity"
                  type="number"
                  min="1"
                  value={newItemQuantity}
                  onChange={(e) => setNewItemQuantity(Number(e.target.value))}
                />
              </div>
              
              <div className="space-y-2">
                <label htmlFor="item-unit" className="text-sm font-medium">
                  {t('shoppingList.addDialog.unitLabel')}
                </label>
                <Select
                  value={newItemUnit}
                  onValueChange={setNewItemUnit}
                >
                  <SelectTrigger id="item-unit">
                    <SelectValue placeholder={t('shoppingList.addDialog.unitLabel')} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={t('shoppingList.addDialog.unitItem')}>{t('shoppingList.addDialog.unitItem')}</SelectItem>
                    <SelectItem value={t('shoppingList.addDialog.unitPack')}>{t('shoppingList.addDialog.unitPack')}</SelectItem>
                    <SelectItem value={t('shoppingList.addDialog.unitKg')}>{t('shoppingList.addDialog.unitKg')}</SelectItem>
                    <SelectItem value={t('shoppingList.addDialog.unitLiter')}>{t('shoppingList.addDialog.unitLiter')}</SelectItem>
                    <SelectItem value={t('shoppingList.addDialog.unitOther')}>{t('shoppingList.addDialog.unitOther')}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="item-category" className="text-sm font-medium">
                {t('shoppingList.addDialog.categoryLabel')}
              </label>
              <Select
                value={newItemCategory}
                onValueChange={setNewItemCategory}
              >
                <SelectTrigger id="item-category">
                  <SelectValue placeholder={t('shoppingList.addDialog.categoryPlaceholder')} />
                </SelectTrigger>
                <SelectContent>
                  {categories.map((category) => (
                    <SelectItem key={category.id} value={category.id}>
                      {translateCategory(t, category.id)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsAddDialogOpen(false)}
            >
              {t('shoppingList.addDialog.cancel')}
            </Button>
            <Button
              onClick={handleAddItem}
              disabled={!newItemName.trim()}
            >
              {t('shoppingList.addDialog.add')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppLayout>
  );
};

export default ShoppingList;
