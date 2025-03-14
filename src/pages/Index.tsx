
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, SortAsc, RefreshCw } from 'lucide-react';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui-custom/Badge';
import ItemCard from '@/components/ui-custom/ItemCard';
import AppLayout from '@/components/layout/AppLayout';
import { usePantryItems } from '@/hooks/usePantryItems';
import { FilterOption, SortOption } from '@/types';
import { cn } from '@/lib/utils';
import { useToast } from '@/components/ui/use-toast';

const PantryPage: React.FC = () => {
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState('');
  const {
    items,
    categories,
    isLoading,
    filter,
    setFilter,
    sortBy,
    setSortBy,
    addToShoppingList,
  } = usePantryItems();

  const filterOptions: { value: FilterOption; label: string }[] = [
    { value: 'all', label: 'All Items' },
    { value: 'expiring', label: 'Expiring Soon' },
    { value: 'expired', label: 'Expired' },
    { value: 'finished', label: 'Out of Stock' },
  ];

  const sortOptions: { value: SortOption; label: string }[] = [
    { value: 'name', label: 'Name' },
    { value: 'expirationDate', label: 'Expiration Date' },
    { value: 'purchaseDate', label: 'Purchase Date' },
    { value: 'category', label: 'Category' },
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
        name: 'Unknown',
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

  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-col md:flex-row justify-between items-center mb-6 gap-4">
          <div>
            <h1 className="text-3xl font-medium">My Pantry</h1>
            <p className="text-muted-foreground">
              {filteredItems.length} items in your pantry
            </p>
          </div>
          
          <div className="relative w-full md:w-auto">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search items..."
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
              <SelectValue placeholder="Filter" />
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
              <SelectValue placeholder="Sort by" />
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
                  title: "Coming Soon",
                  description: "Category filtering will be available soon",
                });
              }}
            >
              {category.name}
            </Badge>
          ))}
        </div>

        {isLoading ? (
          <div className="flex justify-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : filteredItems.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-12"
          >
            <h3 className="text-xl font-medium mb-2">No items found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery
                ? "No items match your search criteria"
                : filter !== 'all'
                ? "No items match the selected filter"
                : "Add your first pantry item to get started"}
            </p>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredItems.map((item) => (
              <ItemCard
                key={item.id}
                item={item}
                category={getCategoryForItem(item.categoryId)}
                onAddToShoppingList={() => addToShoppingList(item)}
                onView={() => {
                  // View details (will implement in future)
                  toast({
                    title: "Coming Soon",
                    description: "Item details view will be available soon",
                  });
                }}
              />
            ))}
          </div>
        )}
      </motion.div>
    </AppLayout>
  );
};

export default PantryPage;
