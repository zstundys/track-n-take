import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, SortAsc, RefreshCw, ShoppingBag, PlusCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui-custom/Badge';
import ItemCard from '@/components/ui-custom/ItemCard';
import EditItemModal from '@/components/ui-custom/EditItemModal';
import AppLayout from '@/components/layout/AppLayout';
import { usePantryItems } from '@/hooks/usePantryItems';
import { FilterOption, SortOption, PantryItem } from '@/types';
import { cn } from '@/lib/utils';
import { useToast } from '@/components/ui/use-toast';
import { translateCategory } from '@/utils/categoryTranslation';

const PantryPage: React.FC = () => {
  const { toast } = useToast();
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedItem, setSelectedItem] = useState<PantryItem | null>(null);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const {
    items,
    categories,
    isLoading,
    filter,
    setFilter,
    sortBy,
    setSortBy,
    addToShoppingList,
    updateItem,
    deleteItem,
  } = usePantryItems();

  const filterOptions: { value: FilterOption; label: string }[] = [
    { value: 'all', label: t('pantry.filter.all') },
    { value: 'expiring', label: t('pantry.filter.expiring') },
    { value: 'expired', label: t('pantry.filter.expired') },
    { value: 'finished', label: t('pantry.filter.finished') },
  ];

  const sortOptions: { value: SortOption; label: string }[] = [
    { value: 'name', label: t('pantry.sort.name') },
    { value: 'expirationDate', label: t('pantry.sort.expirationDate') },
    { value: 'purchaseDate', label: t('pantry.sort.purchaseDate') },
    { value: 'category', label: t('pantry.sort.category') },
  ];

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

  // Handle filter change
  const handleFilterChange = (value: string) => {
    setFilter(value as FilterOption);
  };

  // Handle sort change
  const handleSortChange = (value: string) => {
    setSortBy(value as SortOption);
  };

  // Handle item click to open edit modal
  const handleItemClick = (item: PantryItem) => {
    setSelectedItem(item);
    setIsEditModalOpen(true);
  };

  // Handle closing the edit modal
  const handleCloseEditModal = () => {
    setIsEditModalOpen(false);
    setSelectedItem(null);
  };

  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4">
          <div>
            <h1 className="text-3xl font-medium">{t('pantry.title')}</h1>
            <p className="text-muted-foreground">
              {t('pantry.itemCount', { count: filteredItems.length })}
            </p>
          </div>
          
          <div className="relative w-full md:w-auto">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder={t('pantry.search')}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 w-full md:w-64 rounded-full bg-background border-input"
            />
          </div>
        </div>

        <div className="flex gap-2 mb-6 overflow-x-auto pb-2 hide-scrollbar">
          <Select value={filter} onValueChange={handleFilterChange}>
            <SelectTrigger className="w-[150px] h-9 gap-1">
              <Filter className="h-4 w-4" />
              <SelectValue placeholder={t('pantry.filter.title')} />
            </SelectTrigger>
            <SelectContent>
              {filterOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={sortBy} onValueChange={handleSortChange}>
            <SelectTrigger className="w-[150px] h-9 gap-1">
              <SortAsc className="h-4 w-4" />
              <SelectValue placeholder={t('pantry.sort.title')} />
            </SelectTrigger>
            <SelectContent>
              {sortOptions.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {categories.map((category) => (
            <Badge
              key={category.id}
              variant={category.color as any}
              className="whitespace-nowrap cursor-pointer"
              onClick={() => {
                toast({
                  title: t('common.comingSoon'),
                  description: t('pantry.categoryFilterSoon'),
                });
              }}
            >
              {translateCategory(t, category.id)}
            </Badge>
          ))}
        </div>

        {isLoading ? (
          <div className="flex justify-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin text-primary" />
            <span className="sr-only">{t('pantry.loading')}</span>
          </div>
        ) : filteredItems.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="bg-primary/10 rounded-full p-6 mb-4">
              <ShoppingBag className="h-12 w-12 text-primary" />
            </div>
            <h2 className="text-2xl font-medium mb-2">{t('pantry.empty')}</h2>
            <p className="text-muted-foreground max-w-md mb-6">
              {t('pantry.emptyDescription')}
            </p>
            <Button 
              size="lg" 
              className="gap-2"
              onClick={() => navigate('/add-item')}
            >
              <PlusCircle className="h-5 w-5" />
              <span>{t('pantry.addFirst')}</span>
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredItems.map((item) => (
              <ItemCard
                key={item.id}
                item={item}
                category={getCategoryForItem(item.categoryId)}
                onAddToShoppingList={() => addToShoppingList(item)}
                onView={() => handleItemClick(item)}
              />
            ))}
          </div>
        )}
      </motion.div>

      {/* Edit Item Modal */}
      <EditItemModal
        item={selectedItem}
        category={selectedItem ? getCategoryForItem(selectedItem.categoryId) : null}
        isOpen={isEditModalOpen}
        onClose={handleCloseEditModal}
        onUpdate={updateItem}
        onAddToShoppingList={addToShoppingList}
        onDelete={async (item) => {
          const success = await deleteItem(item);
          if (success) {
            handleCloseEditModal();
          }
          return success;
        }}
      />
    </AppLayout>
  );
};

export default PantryPage;
