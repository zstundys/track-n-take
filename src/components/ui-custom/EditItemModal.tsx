import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { PantryItem, Category, Unit, translateUnit } from "@/types";
import { translateCategory } from "@/utils/category.hooks";
import { Badge } from "./Badge";
import { Minus, Plus, ShoppingCart, Save, Trash2 } from "lucide-react";

interface EditItemModalProps {
  item: PantryItem | null;
  category: Category | null;
  isOpen: boolean;
  onClose: () => void;
  onUpdate: (id: string, updates: Partial<PantryItem>) => Promise<boolean>;
  onAddToShoppingList: (item: PantryItem) => Promise<boolean>;
  onDelete?: (id: string) => Promise<boolean>;
}

const EditItemModal: React.FC<EditItemModalProps> = ({
  item,
  category,
  isOpen,
  onClose,
  onUpdate,
  onAddToShoppingList,
  onDelete,
}) => {
  const { t } = useTranslation();
  const [quantity, setQuantity] = useState<number>(0);
  const [isUpdating, setIsUpdating] = useState(false);
  const [isAddingToShoppingList, setIsAddingToShoppingList] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (item) {
      setQuantity(item.quantity);
    }
  }, [item]);

  if (!item || !category) return null;

  const handleQuantityChange = (newValue: number[]) => {
    setQuantity(newValue[0]);
  };

  const handleIncrement = () => {
    setQuantity((prev) => Math.min(prev + 1, 100));
  };

  const handleDecrement = () => {
    setQuantity((prev) => Math.max(prev - 1, 0));
  };

  const handleSave = async () => {
    if (!item) return;

    setIsUpdating(true);
    try {
      // If quantity is 0, mark as finished
      const updates: Partial<PantryItem> = {
        quantity,
        isFinished: quantity === 0,
      };

      const success = await onUpdate(item.id, updates);

      if (success) {
        onClose();
      }
    } finally {
      setIsUpdating(false);
    }
  };

  const handleAddToShoppingList = async () => {
    if (!item) return;

    setIsAddingToShoppingList(true);
    try {
      await onAddToShoppingList(item);
    } finally {
      setIsAddingToShoppingList(false);
    }
  };

  const handleDelete = async () => {
    if (!item || !onDelete) return;

    // Show confirmation dialog
    const confirmMessage = t("pantry.edit.deleteConfirmation", {
      name: item.name,
    });
    const isConfirmed = window.confirm(confirmMessage);

    if (!isConfirmed) return;

    setIsDeleting(true);
    try {
      const success = await onDelete(item.id);
      if (success) {
        onClose();
      }
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span>{item.name}</span>
            {category && (
              <Badge variant={category.color as any} className="ml-2">
                {translateCategory(t, category.id)}
              </Badge>
            )}
          </DialogTitle>
        </DialogHeader>

        <div className="py-4">
          <div className="mb-6">
            <Label className="text-base font-medium mb-2 block">
              {t("pantry.itemCard.quantity")}
            </Label>

            <div className="flex items-center gap-4 mb-4">
              <Button
                variant="outline"
                size="icon"
                onClick={handleDecrement}
                disabled={quantity <= 0}
              >
                <Minus className="h-4 w-4" />
              </Button>

              <div className="flex-1">
                <Slider
                  value={[quantity]}
                  min={0}
                  max={50}
                  step={1}
                  onValueChange={handleQuantityChange}
                />
              </div>

              <Button variant="outline" size="icon" onClick={handleIncrement}>
                <Plus className="h-4 w-4" />
              </Button>

              <div className="w-16">
                <Input
                  type="number"
                  value={quantity}
                  onChange={(e) => setQuantity(Number(e.target.value))}
                  min={0}
                  max={100}
                />
              </div>
            </div>

            <div className="text-sm text-muted-foreground">
              <p>
                {t("pantry.edit.currentAmount")}: {quantity}{" "}
                {translateUnit(t, item.unit)}
              </p>
            </div>
          </div>
        </div>

        <DialogFooter className="flex flex-col-reverse sm:flex-row-reverse gap-2 items-baseline">
          <Button
            onClick={handleSave}
            className="w-full sm:w-auto gap-2"
            disabled={isUpdating}
          >
            <Save className="h-4 w-4" />
            <span>{t("common.save")}</span>
          </Button>

          <Button
            variant="outline"
            onClick={handleAddToShoppingList}
            className="w-full sm:w-auto gap-2"
            disabled={isAddingToShoppingList}
          >
            <ShoppingCart className="h-4 w-4" />
            <span>{t("pantry.itemCard.addToShopping")}</span>
          </Button>

          <div className="grow" />

          {onDelete && (
            <Button
              variant="destructive"
              size="icon"
              onClick={handleDelete}
              disabled={isDeleting}
              title={t("common.delete")}
              className="order-last sm:order-none"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditItemModal;
