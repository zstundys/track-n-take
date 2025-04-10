import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { MAX_ITEM_QUANTITY } from "@/constants/limits";
import { useIntl } from "@/hooks/useIntl";
import { usePantryItems } from "@/hooks/usePantryItems";
import { Category, PantryItem } from "@/types";
import { translateCategory } from "@/utils/category.hooks";
import { Save, ShoppingCart } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "./Badge";
import DatePickerWithPresets from "./DatePickerWithPresets";

interface EditItemModalProps {
  item: PantryItem | null;
  category: Category | null;
  isOpen: boolean;
  onClose: () => void;
  onUpdate: (id: string, updates: Partial<PantryItem>) => Promise<boolean>;
  onAddToShoppingList: (item: PantryItem) => Promise<boolean>;
}

const EditItemModal: React.FC<EditItemModalProps> = ({
  item,
  category,
  isOpen,
  onClose,
  onUpdate,
  onAddToShoppingList,
}) => {
  const { t } = useTranslation();
  const api = usePantryItems();
  const [quantity, setQuantity] = useState<number>(0);
  const [expirationDate, setExpirationDate] = useState<Date | undefined>(
    undefined
  );
  const [isUpdating, setIsUpdating] = useState(false);
  const [isAddingToShoppingList, setIsAddingToShoppingList] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (item) {
      setQuantity(item.quantity);
      setExpirationDate(
        item.expirationDate ? new Date(item.expirationDate) : undefined
      );
    }
  }, [item]);

  if (!item || !category) return null;

  const handleQuantityChange = (newValue: number[]) => {
    setQuantity(newValue[0]);
  };

  const handleSave = async () => {
    if (!item) return;

    setIsUpdating(true);
    try {
      // If quantity is 0, mark as finished
      const updates: Partial<PantryItem> = {
        quantity,
        isFinished: quantity === 0,
        expirationDate: expirationDate ? expirationDate.getTime() : null,
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

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()} modal>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span>{item.name}</span>
            {category && (
              <Badge variant={category.color} className="ml-2">
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

            <Slider
              value={[quantity]}
              min={0}
              max={MAX_ITEM_QUANTITY}
              step={1}
              onValueChange={handleQuantityChange}
            />
          </div>

          <div className="mb-6">
            <Label className="text-base font-medium mb-2 block">
              {t("pantry.itemCard.expirationDate")}
            </Label>

            <DatePickerWithPresets
              value={expirationDate}
              onDateChange={setExpirationDate}
              placeholder={t("pantry.edit.pickDate")}
              fromDate={new Date()}
            />
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
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditItemModal;
