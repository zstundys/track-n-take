import React from "react";
import { motion } from "framer-motion";
import { useTranslation } from "react-i18next";
import { Card } from "@/components/ui/card";
import { Badge } from "./Badge";
import { cn } from "@/lib/utils";
import { PantryItem, Category } from "@/types";
import {
  AlertCircle,
  Check,
  AlertTriangle,
  Archive,
  ShoppingCart,
} from "lucide-react";
import { getImage } from "@/utils/imageStorage";
import { translateCategory } from "@/utils/categoryTranslation";

interface ItemCardProps {
  item: PantryItem;
  category: Category;
  onAddToShoppingList?: () => void;
  onView?: () => void;
  className?: string;
}

const ItemCard: React.FC<ItemCardProps> = ({
  item,
  category,
  onAddToShoppingList,
  onView,
  className,
}) => {
  const { t, i18n } = useTranslation();
  const isExpired =
    item.expirationDate && new Date(item.expirationDate) < new Date();
  const isExpiringSoon =
    item.expirationDate &&
    !isExpired &&
    new Date(item.expirationDate) <
      new Date(Date.now() + 3 * 24 * 60 * 60 * 1000);

  const getStatusBadge = () => {
    if (item.isFinished) {
      return (
        <Badge variant="gray" className="gap-1" animated>
          <Archive className="h-3 w-3" />
          <span>{t('pantry.filter.finished')}</span>
        </Badge>
      );
    }

    if (isExpired) {
      return (
        <Badge variant="destructive" className="gap-1" animated>
          <AlertCircle className="h-3 w-3" />
          <span>{t('pantry.itemCard.expired')}</span>
        </Badge>
      );
    }

    if (isExpiringSoon) {
      return (
        <Badge variant="yellow" className="gap-1" animated>
          <AlertTriangle className="h-3 w-3" />
          <span>{t('pantry.filter.expiring')}</span>
        </Badge>
      );
    }

    return (
      <Badge variant="green" className="gap-1">
        <Check className="h-3 w-3" />
        <span>OK</span>
      </Badge>
    );
  };

  const getExpirationText = () => {
    if (!item.expirationDate) return null;

    const expirationDate = new Date(item.expirationDate);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const expirationDay = new Date(expirationDate);
    expirationDay.setHours(0, 0, 0, 0);

    const isToday = expirationDay.getTime() === today.getTime();

    if (isToday) {
      return t('pantry.itemCard.today');
    }

    if (isExpired) {
      const days = Math.abs(
        Math.floor(
          (today.getTime() - expirationDay.getTime()) / (1000 * 60 * 60 * 24)
        )
      );
      return t('pantry.itemCard.daysAgo', { count: days });
    }

    const days = Math.floor(
      (expirationDay.getTime() - today.getTime()) / (1000 * 60 * 60 * 24)
    );
    return t('pantry.itemCard.daysLeft', { count: days });
  };

  // Format date using Intl API
  const formatDate = (date: string | number | Date) => {
    return new Intl.DateTimeFormat(i18n.language, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(new Date(date));
  };

  // Get item image if it exists
  const itemImage = item.image ? getImage(item.image) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ scale: 1.02 }}
      className={className}
    >
      <Card
        className={cn(
          "overflow-hidden transition-all cursor-pointer hover:shadow-md",
          item.isFinished && "opacity-70"
        )}
        onClick={onView}
      >
        {itemImage && (
          <div className="w-full h-32 bg-muted">
            <img 
              src={itemImage} 
              alt={item.name} 
              className="w-full h-full object-cover"
            />
          </div>
        )}
        
        <div className="p-4">
          <div className="flex justify-between items-start">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Badge variant={category.color as any}>{translateCategory(t, category.id)}</Badge>
                {getStatusBadge()}
              </div>
              <h3 className="font-medium text-lg">{item.name}</h3>
            </div>

            {(item.isFinished || isExpired) && (
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="p-1.5 bg-primary/10 hover:bg-primary/20 text-primary rounded-full"
                onClick={(e) => {
                  e.stopPropagation();
                  onAddToShoppingList?.();
                }}
                aria-label="Add to shopping list"
              >
                <ShoppingCart className="h-4 w-4" />
              </motion.button>
            )}
          </div>

          <div className="mt-2 text-sm text-muted-foreground">
            <p>
              {item.quantity} {item.unit}
            </p>

            {item.expirationDate && (
              <div className="text-sm text-muted-foreground mb-2">
                <span>{t('pantry.itemCard.expires')}: </span>
                <span
                  className={cn(
                    isExpired && "text-destructive",
                    isExpiringSoon && "text-amber-500"
                  )}
                >
                  {formatDate(item.expirationDate)}
                  {" "}
                  ({getExpirationText()})
                </span>
              </div>
            )}

            {item.purchaseDate && (
              <p className="mt-1 text-xs">
                Purchased: {formatDate(item.purchaseDate)}
              </p>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default ItemCard;
