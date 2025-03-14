
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
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

const AddItem: React.FC = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
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
      title: "Image Added",
      description: "Photo has been attached to this item"
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
      navigate('/');
    }
  };
  
  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-3xl font-medium mb-6">Add Pantry Item</h1>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid gap-4">
            {/* Image preview (if an image is selected) */}
            {imageId && (
              <div className="relative bg-muted rounded-lg overflow-hidden">
                <img 
                  src={getImage(imageId) || ''} 
                  alt="Item" 
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
                Item Name <span className="text-destructive">*</span>
              </label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter item name"
                required
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label htmlFor="quantity" className="text-sm font-medium">
                  Quantity
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
                  Unit
                </label>
                <Input
                  id="unit"
                  value={unit}
                  onChange={(e) => setUnit(e.target.value)}
                  placeholder="e.g., pack, bottle, lb"
                />
              </div>
            </div>
            
            <div className="space-y-2">
              <label htmlFor="category" className="text-sm font-medium">
                Category <span className="text-destructive">*</span>
              </label>
              <Select
                value={categoryId}
                onValueChange={setCategoryId}
                required
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
            
            <div className="space-y-2">
              <label htmlFor="expiration-date" className="text-sm font-medium">
                Expiration Date
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
                    {expirationDate ? format(expirationDate, "PPP") : "Select expiration date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
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
                Purchase Date
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
                    {purchaseDate ? format(purchaseDate, "PPP") : "Select purchase date"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
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
                Notes
              </label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add any additional notes here"
                rows={3}
              />
            </div>
            
            {!showCamera && !imageId && (
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1 gap-2"
                  onClick={() => {
                    toast({
                      title: "Coming Soon",
                      description: "Barcode scanning will be available in a future update",
                    });
                  }}
                >
                  <Scan className="h-4 w-4" />
                  <span>Scan Barcode</span>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1 gap-2"
                  onClick={() => setShowCamera(true)}
                >
                  <ImageIcon className="h-4 w-4" />
                  <span>Add Photo</span>
                </Button>
              </div>
            )}
          </div>
          
          <div className="flex gap-4">
            <Button
              type="button"
              variant="outline"
              className="flex-1"
              onClick={() => navigate('/')}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              className="flex-1"
              disabled={!isFormValid}
            >
              Add Item
            </Button>
          </div>
        </form>
      </motion.div>
    </AppLayout>
  );
};

export default AddItem;
