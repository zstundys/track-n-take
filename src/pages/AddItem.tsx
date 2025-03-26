import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { format } from 'date-fns';
import { Calendar as CalendarIcon, Scan, Image as ImageIcon, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Calendar } from '@/components/ui/calendar';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import AppLayout from '@/components/layout/AppLayout';
import { usePantryItems } from '@/hooks/usePantryItems';
import { cn } from '@/lib/utils';
import { useToast } from '@/components/ui/use-toast';
import CameraCapture from '@/components/ui-custom/CameraCapture';
import { getImage, deleteImage } from '@/utils/imageStorage';
import { translateCategory } from '@/utils/categoryTranslation';

const AddItem: React.FC = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { categories, addItem } = usePantryItems();
  
  // Form state
  const [name, setName] = useState('');
  const [quantity, setQuantity] = useState(1);
  const [unit, setUnit] = useState('item');
  const [categoryId, setCategoryId] = useState('');
  const [expirationDate, setExpirationDate] = useState<Date | undefined>(undefined);
  const [purchaseDate, setPurchaseDate] = useState<Date | undefined>(new Date());
  const [notes, setNotes] = useState('');
  const [imageId, setImageId] = useState<string | null>(null);
  const [showCamera, setShowCamera] = useState(false);
  
  // Form validation
  const isFormValid = name.trim() !== '' && categoryId !== '';
  
  // Handle image capture
  const handleImageCaptured = (capturedImageId: string) => {
    setImageId(capturedImageId);
    setShowCamera(false);
    
    toast({
      title: t('addItem.imageAdded'),
      description: t('addItem.imageAttached')
    });
  };
  
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
      categoryId,
      expirationDate: expirationDate ? expirationDate.getTime() : null,
      purchaseDate: purchaseDate ? purchaseDate.getTime() : null,
      notes,
      isFinished: false,
      image: imageId || undefined,
    });
    
    if (success) {
      toast({
        title: t('addItem.form.success'),
        description: name
      });
      navigate('/');
    } else {
      toast({
        title: t('addItem.form.error'),
        variant: "destructive"
      });
    }
  };
  
  // Format date for display
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat(navigator.language, {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(date);
  };
  
  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-medium mb-2">{t('addItem.title')}</h1>
        <p className="text-muted-foreground mb-6">{t('addItem.subtitle')}</p>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid gap-4">
            {/* Image preview (if an image is selected) */}
            {imageId && (
              <div className="relative bg-muted rounded-lg overflow-hidden">
                <img 
                  src={getImage(imageId) || ''} 
                  alt={t('addItem.form.itemImage')} 
                  className="w-full h-auto object-contain max-h-64"
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
            
            {/* Camera capture component */}
            {showCamera && (
              <div className="space-y-2">
                <CameraCapture onImageCaptured={handleImageCaptured} />
              </div>
            )}
            
            <div className="space-y-2">
              <label htmlFor="name" className="text-sm font-medium">
                {t('addItem.form.nameLabel')} <span className="text-destructive">*</span>
              </label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t('addItem.form.namePlaceholder')}
                required
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label htmlFor="quantity" className="text-sm font-medium">
                  {t('addItem.form.quantityLabel')}
                </label>
                <Input
                  id="quantity"
                  type="number"
                  min="0"
                  step="0.1"
                  value={quantity}
                  onChange={(e) => setQuantity(Number(e.target.value))}
                />
              </div>
              <div className="space-y-2">
                <label htmlFor="unit" className="text-sm font-medium">
                  {t('addItem.form.unitLabel')}
                </label>
                <Select
                  value={unit}
                  onValueChange={setUnit}
                >
                  <SelectTrigger id="unit">
                    <SelectValue placeholder={t('addItem.form.unitLabel')} />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={"item"}>{t('addItem.form.unitItem', { count: quantity }  )}</SelectItem>
                    <SelectItem value={"pack"}>{t('addItem.form.unitPack', { count: quantity })}</SelectItem>
                    <SelectItem value={"kg"}>{t('addItem.form.unitKg', { count: quantity })}</SelectItem>
                    <SelectItem value={"liter"}>{t('addItem.form.unitLiter', { count: quantity })}</SelectItem>
                    <SelectItem value={"other"}>{t('addItem.form.unitOther', { count: quantity })}</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="category" className="text-sm font-medium">
                {t('addItem.form.categoryLabel')} <span className="text-destructive">*</span>
              </label>
              <Select
                value={categoryId}
                onValueChange={setCategoryId}
                required
              >
                <SelectTrigger id="category">
                  <SelectValue placeholder={t('addItem.form.categoryPlaceholder')} />
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
            
            <div className="space-y-2">
              <label htmlFor="expiration-date" className="text-sm font-medium">
                {t('addItem.form.expirationLabel')}
              </label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    id="expiration-date"
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !expirationDate && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {expirationDate ? (
                      formatDate(expirationDate)
                    ) : (
                      <span>{t('addItem.form.expirationPlaceholder')}</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={expirationDate}
                    onSelect={setExpirationDate}
                    initialFocus
                  />
                </PopoverContent>
              </Popover>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="purchase-date" className="text-sm font-medium">
                {t('addItem.form.purchaseDateLabel')}
              </label>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    id="purchase-date"
                    variant="outline"
                    className={cn(
                      "w-full justify-start text-left font-normal",
                      !purchaseDate && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4" />
                    {purchaseDate ? (
                      formatDate(purchaseDate)
                    ) : (
                      <span>{t('addItem.form.purchaseDatePlaceholder')}</span>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0">
                  <Calendar
                    mode="single"
                    selected={purchaseDate}
                    onSelect={setPurchaseDate}
                    initialFocus
                  />
                </PopoverContent>
              </Popover>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="notes" className="text-sm font-medium">
                {t('addItem.form.notesLabel')}
              </label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder={t('addItem.form.notesPlaceholder')}
                className="min-h-[100px]"
              />
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-2 pt-4">
            <Button 
              type="button" 
              variant="outline" 
              className="flex-1 gap-2"
              onClick={() => setShowCamera(!showCamera)}
            >
              {showCamera ? (
                <>
                  <X className="h-4 w-4" />
                  <span>{t('addItem.form.cancelPhoto')}</span>
                </>
              ) : (
                <>
                  <ImageIcon className="h-4 w-4" />
                  <span>{t('addItem.form.takePhoto')}</span>
                </>
              )}
            </Button>
            
            <Button 
              type="submit" 
              className="flex-1 gap-2"
              disabled={!isFormValid}
            >
              <span>{t('addItem.form.submit')}</span>
            </Button>
          </div>
        </form>
      </motion.div>
    </AppLayout>
  );
};

export default AddItem;
