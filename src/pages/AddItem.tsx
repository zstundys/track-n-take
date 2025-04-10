import React, {
  useState,
  useEffect,
  useId,
  ComponentProps,
  useRef,
} from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { format } from "date-fns";
import {
  Calendar as CalendarIcon,
  Scan,
  Image as ImageIcon,
  X,
  Plus,
  Minus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Calendar } from "@/components/ui/calendar";
import { Slider } from "@/components/ui/slider";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import AppLayout from "@/components/layout/AppLayout";
import { usePantryItems } from "@/hooks/usePantryItems";
import { cn } from "@/lib/utils";
import { useToast } from "@/components/ui/use-toast";
import CameraCapture from "@/components/ui-custom/CameraCapture";
import DatePickerWithPresets from "@/components/ui-custom/DatePickerWithPresets";
import { getImage, deleteImage, getImageAsBlob } from "@/utils/imageStorage";
import {
  translateCategory,
  getCategoryColorClasses,
} from "@/utils/category.hooks";

import styles from "./AddItem.module.css";
import { useIntl } from "@/hooks/useIntl";
import { useImageRecognizerWorker } from "@/hooks/use-image-recognizer-worker";
import { CategoryId } from "@/types";

const NOW = new Date();
const SIX_MONTHS_FROM_NOW = new Date();
SIX_MONTHS_FROM_NOW.setDate(NOW.getDate() + 180);

const AddItem: React.FC = () => {
  const navigate = useNavigate();
  const { shortDate } = useIntl();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { categories, addItem } = usePantryItems();
  // Form state
  const [name, setName] = useState("");
  const [quantity, setQuantity] = useState(1);
  const [unit, setUnit] = useState("item");
  const [categoryId, setCategoryId] = useState<CategoryId | "">("");
  const [expirationDate, setExpirationDate] = useState<Date | undefined>(
    undefined
  );
  const { categorizeImage, isThinking } = useImageRecognizerWorker();
  const [notes, setNotes] = useState("");
  const [imageId, setImageId] = useState<string | null>(null);

  const isFormValid = name.trim() !== "";

  // Handle image capture
  const handleImageCaptured = (capturedImageId: string) => {
    setImageId(capturedImageId);

    toast({
      title: t("addItem.imageAdded"),
      description: t("addItem.imageAttached"),
    });
  };

  const imageBlob = imageId ? getImageAsBlob(imageId) : undefined;
  const isLoadingRef = useRef(false);

  useEffect(() => {
    if (imageBlob && !isThinking && !isLoadingRef.current) {
      isLoadingRef.current = true;

      document.documentElement.classList.add("loading");

      categorizeImage(imageBlob)
        .then((result) => {
          if (result) {
            const bestMatch = result.classification[0].label as CategoryId;

            setCategoryId(bestMatch);
            setName(result.description?.generated_text ?? "");
          }

          document.documentElement.classList.remove("loading");
          isLoadingRef.current = false;
        })
        .catch(() => {
          document.documentElement.classList.remove("loading");
          isLoadingRef.current = false;
        });
    }
  }, [categorizeImage, imageBlob, isThinking]);

  // Remove image
  const handleRemoveImage = () => {
    if (imageId) {
      deleteImage(imageId);
      setImageId(null);
    }
  };

  // Submit form
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isFormValid) return;

    const success = await addItem({
      name,
      quantity,
      unit,
      categoryId: categoryId || null,
      expirationDate: expirationDate ? expirationDate.getTime() : null,
      notes,
      isFinished: false,
      image: imageId || undefined,
    });

    if (success) {
      toast({
        title: t("addItem.form.success"),
        description: name,
      });
      navigate("/");
    } else {
      toast({
        title: t("addItem.form.error"),
        variant: "destructive",
      });
    }
  };

  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-medium mb-2">{t("addItem.title")}</h1>
        <p className="text-muted-foreground mb-6">{t("addItem.subtitle")}</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid gap-4">
            {!imageId && (
              <div className="space-y-2">
                <CameraCapture onImageCaptured={handleImageCaptured} />
              </div>
            )}

            {/* Image preview (if an image is selected) */}
            {imageId && (
              <div className="relative bg-muted rounded-lg overflow-hidden">
                <img
                  src={getImage(imageId) || ""}
                  aria-hidden="true"
                  className="object-cover absolute inset-0 w-full h-full pointer-events-none blur-3xl opacity-30"
                />
                <img
                  src={getImage(imageId) || ""}
                  alt={t("addItem.form.itemImage")}
                  className="relative w-full h-auto object-contain max-h-96"
                />
                <Button
                  type="button"
                  variant="destructive"
                  size="icon"
                  className="absolute top-2 right-2 rounded-full opacity-90 hover:opacity-100"
                  onClick={handleRemoveImage}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            )}

            <div className="space-y-2">
              <label htmlFor="name" className="text-sm font-medium">
                {t("addItem.form.nameLabel")}{" "}
                <span className="text-destructive">*</span>
              </label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("addItem.form.namePlaceholder")}
                required
                autoFocus
              />
            </div>

            <div className="grid grid-cols-[1fr,auto] gap-4">
              <div className="space-y-2">
                <label htmlFor="quantity" className="text-sm font-medium">
                  {t("addItem.form.quantityLabel")}
                </label>
                <Slider
                  value={[quantity]}
                  min={0}
                  max={20}
                  step={1}
                  onValueChange={(newValue) => setQuantity(newValue[0])}
                />
              </div>
              <div className="space-y-2 max-w-max">
                <label htmlFor="unit" className="text-sm font-medium">
                  {t("addItem.form.unitLabel")}
                </label>
                <Select value={unit} onValueChange={setUnit}>
                  <SelectTrigger id="unit">
                    <SelectValue placeholder={t("addItem.form.unitLabel")} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={"item"}>
                      {t("addItem.form.unitItem", { count: quantity })}
                    </SelectItem>
                    <SelectItem value={"pack"}>
                      {t("addItem.form.unitPack", { count: quantity })}
                    </SelectItem>
                    <SelectItem value={"kg"}>
                      {t("addItem.form.unitKg", { count: quantity })}
                    </SelectItem>
                    <SelectItem value={"liter"}>
                      {t("addItem.form.unitLiter", { count: quantity })}
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2" id="terminalSelection">
              <label htmlFor="category" className="text-sm font-medium">
                {t("addItem.form.categoryLabel")}
              </label>
              <div className="flex flex-wrap gap-2">
                {categories.map((category) => (
                  <label
                    key={category.id}
                    htmlFor={`category-${category.id}`}
                    className={cn(
                      "flex items-center  justify-center px-3 py-1.5 rounded-full border-2 text-sm font-medium cursor-pointer transition-all",
                      getCategoryColorClasses(category.color),
                      category.id === categoryId
                        ? styles.selectedCategory
                        : undefined
                    )}
                  >
                    <input
                      type="radio"
                      name="category"
                      id={`category-${category.id}`}
                      value={category.id}
                      checked={category.id === categoryId}
                      onChange={() => setCategoryId(category.id as CategoryId)}
                      className="sr-only"
                    />
                    {translateCategory(t, category.id)}
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <label htmlFor="expirationDate" className="text-sm font-medium">
                {t("addItem.form.expirationLabel")}
              </label>
              <DatePickerWithPresets
                value={expirationDate}
                onDateChange={setExpirationDate}
                placeholder={t("addItem.form.expirationPlaceholder")}
                fromDate={NOW}
                toDate={SIX_MONTHS_FROM_NOW}
              />
            </div>

            <span className="dark:bg-green-800"></span>

            <div className="space-y-2">
              <label htmlFor="notes" className="text-sm font-medium">
                {t("addItem.form.notesLabel")}
              </label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder={t("addItem.form.notesPlaceholder")}
                className="min-h-[100px]"
              />
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2 pt-4 w-fit mx-auto sticky bottom-32">
            <Button
              type="submit"
              className="flex-1 gap-2 shadow-lg disabled:shadow-none shadow-green-700 dark:shadow-green-900 rounded-full px-8 relative disabled:bg-primary/40 backdrop-blur-[1px] opacity-[revert!important] "
              disabled={!isFormValid}
            >
              <span>{t("addItem.form.submit")}</span>
            </Button>
          </div>
        </form>
      </motion.div>
    </AppLayout>
  );
};

export default AddItem;
