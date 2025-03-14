
import React, { useRef, useState, useCallback } from 'react';
import { Camera, X, Image as ImageIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { saveImage } from '@/utils/imageStorage';

interface CameraCaptureProps {
  onImageCaptured: (imageId: string) => void;
  className?: string;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onImageCaptured, className }) => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera on mobile
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCapturing(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Could not access camera. Please check permissions and try again.');
    }
  }, []);
  
  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsCapturing(false);
  }, []);
  
  // Capture image
  const captureImage = useCallback(() => {
    if (!videoRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    
    const dataURL = canvas.toDataURL('image/jpeg');
    setPreviewImage(dataURL);
    stopCamera();
  }, [stopCamera]);
  
  // Save captured image
  const saveCapture = useCallback(async () => {
    if (!previewImage) return;
    
    try {
      const blob = await fetch(previewImage).then(r => r.blob());
      const imageId = await saveImage(blob);
      onImageCaptured(imageId);
      setPreviewImage(null);
    } catch (error) {
      console.error('Error saving captured image:', error);
      alert('Failed to save image. Please try again.');
    }
  }, [previewImage, onImageCaptured]);
  
  // Upload image from file
  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    try {
      const imageId = await saveImage(file);
      onImageCaptured(imageId);
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Failed to upload image. Please try again.');
    }
    
    // Reset input value to allow the same file to be selected again
    event.target.value = '';
  }, [onImageCaptured]);
  
  // Cancel capture
  const cancelCapture = useCallback(() => {
    setPreviewImage(null);
    if (isCapturing) {
      stopCamera();
    }
  }, [isCapturing, stopCamera]);
  
  return (
    <div className={cn('space-y-4', className)}>
      {isCapturing ? (
        <div className="relative bg-black rounded-lg overflow-hidden">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            className="w-full h-full aspect-video object-cover"
          />
          
          <div className="absolute bottom-0 left-0 right-0 flex justify-center p-4 gap-2 bg-gradient-to-t from-black/80 to-transparent">
            <Button 
              variant="destructive" 
              size="icon" 
              onClick={stopCamera}
              className="rounded-full"
            >
              <X className="h-5 w-5" />
            </Button>
            <Button 
              variant="default" 
              size="icon" 
              onClick={captureImage}
              className="rounded-full"
            >
              <Camera className="h-5 w-5" />
            </Button>
          </div>
        </div>
      ) : previewImage ? (
        <div className="relative bg-black rounded-lg overflow-hidden">
          <img 
            src={previewImage} 
            alt="Preview" 
            className="w-full h-full aspect-video object-contain"
          />
          
          <div className="absolute bottom-0 left-0 right-0 flex justify-center p-4 gap-2 bg-gradient-to-t from-black/80 to-transparent">
            <Button 
              variant="destructive" 
              size="sm"
              onClick={cancelCapture}
            >
              Cancel
            </Button>
            <Button 
              variant="default" 
              size="sm"
              onClick={saveCapture}
            >
              Use Photo
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            className="flex-1 gap-2"
            onClick={startCamera}
          >
            <Camera className="h-4 w-4" />
            <span>Take Photo</span>
          </Button>
          
          <div className="relative flex-1">
            <Button
              type="button"
              variant="outline"
              className="w-full gap-2"
            >
              <ImageIcon className="h-4 w-4" />
              <span>Upload Image</span>
            </Button>
            <input
              type="file"
              accept="image/*"
              className="absolute inset-0 opacity-0 cursor-pointer"
              onChange={handleFileUpload}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraCapture;
