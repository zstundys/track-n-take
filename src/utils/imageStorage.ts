/**
 * Utility functions for storing and retrieving images from local storage
 */

// Max image size to store (in bytes) - default 5MB
const MAX_IMAGE_SIZE = 5 * 1024 * 1024;

// Convert image data URL to blob
const dataURLToBlob = (dataURL: string): Blob => {
  const arr = dataURL.split(",");
  const mime = arr[0].match(/:(.*?);/)?.[1] || "image/png";
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }

  return new Blob([u8arr], { type: mime });
};

// Resize image to reduce storage size
const resizeImage = (file: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(img.src);

      // Calculate new dimensions (max 800px width/height)
      let width = img.width;
      let height = img.height;
      const maxDimension = 800;

      if (width > height && width > maxDimension) {
        height = Math.round((height * maxDimension) / width);
        width = maxDimension;
      } else if (height > maxDimension) {
        width = Math.round((width * maxDimension) / height);
        height = maxDimension;
      }

      // Create canvas and resize
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Could not get canvas context"));
        return;
      }

      // Draw image on canvas
      ctx.drawImage(img, 0, 0, width, height);

      // Convert to data URL with reduced quality
      const dataURL = canvas.toDataURL("image/jpeg", 0.8);
      resolve(dataURL);
    };

    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error("Failed to load image"));
    };
  });
};

// Save image to local storage
export const saveImage = async (imageFile: File | Blob): Promise<string> => {
  try {
    // Resize image if it's too large
    const dataURL = await resizeImage(imageFile);

    // Generate a unique ID for the image
    const imageId = `img_${Date.now()}_${Math.random()
      .toString(36)
      .substring(2, 9)}`;

    // Store in sessionStorage
    sessionStorage.setItem(imageId, dataURL);

    return imageId;
  } catch (error) {
    console.error("Error saving image:", error);
    throw new Error("Failed to save image");
  }
};

// Get image from local storage
export const getImage = (imageId: string): string | null => {
  return sessionStorage.getItem(imageId);
};

export const getImageAsBlob = (imageId: string): Blob | null => {
  const dataURL = sessionStorage.getItem(imageId);
  if (!dataURL) return null;

  try {
    return dataURLToBlob(dataURL);
  } catch (error) {
    console.error("Error converting image to blob:", error);
    return null;
  }
};
// Delete image from local storage
export const deleteImage = (imageId: string): boolean => {
  if (sessionStorage.getItem(imageId)) {
    sessionStorage.removeItem(imageId);
    return true;
  }
  return false;
};
