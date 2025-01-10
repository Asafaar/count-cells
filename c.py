import os
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import label
import imagej
import scyjava

def process_with_imagej(image_path):
    """
    Uses ImageJ to perform segmentation and returns binary mask
    """
    # Initialize ImageJ
    ij = imagej.init('C://Users//asaf9//Downloads//Fiji.app')
    
    # Open image
    image = ij.io().open(image_path)
    
    # Convert to 8-bit
    image = ij.op().convert().uint8(image)
    
    # Auto-threshold using Otsu's method
    thresh = ij.op().threshold().otsu(image)
    mask = ij.op().create().img(image)
    ij.op().threshold().apply(mask, image, thresh)
    
    # Convert to numpy array
    mask_array = ij.py.from_java(mask)
    
    return mask_array > 0

def mask_to_bounding_boxes(binary_mask):
    """
    Converts binary mask to bounding boxes
    """
    # Label connected components
    labeled_mask, num_features = label(binary_mask)
    
    boxes = []
    for i in range(1, num_features + 1):
        # Get coordinates of each cell
        y, x = np.where(labeled_mask == i)
        if len(x) == 0 or len(y) == 0:
            continue
            
        # Calculate bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Convert to YOLO format (x_center, y_center, width, height)
        width = binary_mask.shape[1]
        height = binary_mask.shape[0]
        
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        
        boxes.append([0, x_center, y_center, box_width, box_height])  # 0 is class id for cell
        
    return boxes

def create_yolo_annotations(input_dir, output_dir):
    """
    Process all images in directory and create YOLO annotations
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    for img_name in os.listdir(input_dir):
        if img_name.endswith(('.png', '.jpg', '.tiff')):
            img_path = os.path.join(input_dir, img_name)
            
            # Process with ImageJ
            binary_mask = process_with_imagej(img_path)
            
            # Get bounding boxes
            boxes = mask_to_bounding_boxes(binary_mask)
            
            # Save YOLO annotation
            basename = os.path.splitext(img_name)[0]
            label_path = os.path.join(output_dir, 'labels', f'{basename}.txt')
            
            with open(label_path, 'w') as f:
                for box in boxes:
                    f.write(' '.join(map(str, box)) + '\n')
            
            # Copy image to output directory
            image = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_dir, 'images', img_name), image)

# Example usage
if __name__ == "__main__":
    create_yolo_annotations(
        input_dir='./a',
        output_dir='dataset/train'
    )