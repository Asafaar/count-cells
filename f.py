import cv2
import numpy as np
import os
from scipy import ndimage
import tifffile
import matplotlib.pyplot as plt


def improve_segmentation(mask):
    """
    Improve segmentation by separating touching cells using Watershed algorithm.
    """
    # Ensure binary mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # Distance transform and thresholding for foreground
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Background and unknown regions
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed to refine segmentation
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    # Return improved binary mask
    return (markers > 1).astype(np.uint8) * 255


def create_cell_boxes(mask, min_size=50, max_size=10000):
    """
    Create bounding boxes for cells with size filtering.
    """
    labeled_mask, num_objects = ndimage.label(mask)
    boxes = []
    for obj_idx in range(1, num_objects + 1):
        # Get object coordinates
        y, x = np.where(labeled_mask == obj_idx)

        if len(x) == 0 or len(y) == 0:
            continue

        # Calculate area
        area = len(x)
        if area < min_size or area > max_size:
            continue

        # Get bounding box coordinates
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        boxes.append((x_min, y_min, x_max, y_max))

    return boxes


def weka_tif_to_yolo(segmentation_path, original_image_path, output_dir):
    """
    Convert segmentation mask to YOLO format annotations.
    """
    # Read images
    mask = tifffile.imread(segmentation_path)
    original = tifffile.imread(original_image_path)

    # Ensure binary and improve segmentation
    if len(mask.shape) > 2:
        mask = (mask == 1).astype(np.uint8) * 255

    improved_mask = improve_segmentation(mask)

    # Get dimensions
    height, width = mask.shape

    # Generate bounding boxes
    boxes = create_cell_boxes(improved_mask)

    # Convert to YOLO format
    yolo_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height
        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height
        yolo_boxes.append([0, x_center, y_center, box_width, box_height])

    # Save YOLO annotations
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    with open(os.path.join(output_dir, 'labels', f'{base_name}.txt'), 'w') as f:
        for box in yolo_boxes:
            f.write(' '.join(map(str, box)) + '\n')

    # Save visualization
    vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original.copy()
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    os.makedirs(os.path.join(output_dir, 'visualization'), exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'visualization', f'{base_name}_boxes.jpg'), vis_img)

    return len(boxes)


def process_directory(seg_dir, img_dir, output_dir):
    """
    Process all segmentation and original images in the directory.
    """
    total_cells = 0
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.TIF', '.tif')):
            seg_path = os.path.join(seg_dir, img_name)
            img_path = os.path.join(img_dir, img_name)

            if os.path.exists(seg_path):
                num_cells = weka_tif_to_yolo(seg_path, img_path, output_dir)
                total_cells += num_cells
                print(f"Processed {img_name}: {num_cells} cells detected.")
            else:
                print(f"Segmentation not found for {img_name}.")

    print(f"Total cells detected: {total_cells}")


if __name__ == "__main__":
    # Update paths as needed
    process_directory(
        seg_dir='./cell_dataset/weka',  # Segmentation folder
        img_dir='./cell_dataset/original',  # Original images folder
        output_dir='./yolo_dataset'  # YOLO dataset output
    )
