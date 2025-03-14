
import React from "react";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "./Badge";
import { cn } from "@/lib/utils";
import { PantryItem, Category } from "@/types";
import { formatDistanceToNow, isAfter, isBefore, format } from "date-fns";
import {
  AlertCircle,
  Check,
  AlertTriangle,
  Archive,
  ShoppingCart,
} from "lucide-react";
import { getImage } from "@/utils/imageStorage";

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
  const isExpired =
    item.expirationDate && isBefore(new Date(item.expirationDate), new Date());
  const isExpiringSoon =
    item.expirationDate &&
    !isExpired &&
    isBefore(
      new Date(item.expirationDate),
      new Date(Date.now() + 3 * 24 * 60 * 60 * 1000)
    );

  const getStatusBadge = () => {
    if (item.isFinished) {
      return (
        <Badge variant="gray" className="gap-1" animated>
          <Archive className="h-3 w-3" />
          <span>Out of stock</span>
        </Badge>
      );
    }

    if (isExpired) {
      return (
        <Badge variant="red" className="gap-1" animated>
          <AlertCircle className="h-3 w-3" />
          <span>Expired</span>
        </Badge>
      );
    }

    if (isExpiringSoon) {
      return (
        <Badge variant="yellow" className="gap-1" animated>
          <AlertTriangle className="h-3 w-3" />
          <span>Expiring soon</span>
        </Badge>
      );
    }

    return (
      <Badge variant="green" className="gap-1" animated>
        <Check className="h-3 w-3" />
        <span>Good</span>
      </Badge>
    );
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
                <Badge variant={category.color as any}>{category.name}</Badge>
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
              <p>
                {isExpired
                  ? `Expired ${formatDistanceToNow(
                      new Date(item.expirationDate),
                      { addSuffix: true }
                    )}`
                  : `Expires ${formatDistanceToNow(
                      new Date(item.expirationDate),
                      { addSuffix: true }
                    )}`}
              </p>
            )}

            {item.purchaseDate && (
              <p className="mt-1 text-xs">
                Purchased: {format(new Date(item.purchaseDate), "MMM d, yyyy")}
              </p>
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default ItemCard;
