import cv2
import numpy as np
import os
from scipy import ndimage

def weka_to_yolo(segmentation_path, original_image_path, output_dir):
    """
    Convert Weka segmentation output to YOLO format annotations
    
    Args:
        segmentation_path: Path to the binary mask from Weka
        original_image_path: Path to the original image
        output_dir: Directory to save YOLO annotations and processed images
    """
    # Read images
    mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    original = cv2.imread(original_image_path)
    
    # Make sure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Label connected components
    labeled_mask, num_objects = ndimage.label(mask)
    
    # Get image dimensions
    height, width = mask.shape
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Generate YOLO annotations
    boxes = []
    for obj_idx in range(1, num_objects + 1):
        # Get object coordinates
        y, x = np.where(labeled_mask == obj_idx)
        
        if len(x) == 0 or len(y) == 0:
            continue
            
        # Calculate bounding box
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        
        # Add box [class_id, x_center, y_center, width, height]
        boxes.append([0, x_center, y_center, box_width, box_height])
    
    # Save annotations
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    # Save YOLO format annotations
    with open(os.path.join(output_dir, 'labels', f'{base_name}.txt'), 'w') as f:
        for box in boxes:
            f.write(' '.join(map(str, box)) + '\n')
    
    # Copy original image
    cv2.imwrite(os.path.join(output_dir, 'images', f'{base_name}.jpg'), original)
    
    return len(boxes)

# Example usage
if __name__ == "__main__":
    # Create conversion for a single image
    num_cells = weka_to_yolo(
        segmentation_path='weka_output.png',
        original_image_path='original_image.jpg',
        output_dir='yolo_dataset'
    )
    print(f'Found and annotated {num_cells} cells')