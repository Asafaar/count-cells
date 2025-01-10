import cv2
import numpy as np
import os

def convert_and_visualize_on_original(input_folder, original_images_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.tif', '.jpg', '.TIF')):
            classified_image_path = os.path.join(input_folder, file_name)
            original_image_path = os.path.join(original_images_folder, file_name)
            annotation_file = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".txt")

            classified_image = cv2.imread(classified_image_path, cv2.IMREAD_GRAYSCALE)
            original_image = cv2.imread(original_image_path)

            if classified_image is None or original_image is None:
                print(f"Error loading images for {file_name}. Skipping...")
                continue

            img_height, img_width = classified_image.shape[:2]
            total_cell_count = 0
            all_contours = []

            with open(annotation_file, "w") as f:
                # שינוי חשוב: לולאה רק על מחלקות שאינן רקע
                for class_id in np.unique(classified_image):
                    if class_id == 188:  # הנחה: 188 מייצג את הרקע
                        continue

                    binary_mask = (classified_image == class_id).astype(np.uint8)

                    # פעולות מורפולוגיות (כמו קודם)
                    kernel = np.ones((3,3),np.uint8)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    print(f"Found {len(contours)} contours for class {class_id}")

                    for contour in contours:
                        if cv2.contourArea(contour) < 20:
                            continue

                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            print("Contour with zero area detected. Skipping...")
                            continue

                        x, y, w, h = cv2.boundingRect(contour)
                        x_center = cX / img_width
                        y_center = cY / img_height
                        width = w / img_width
                        height = h / img_height

                        f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                        cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(original_image, (cX, cY), 5, (0, 0, 255), -1)
                        all_contours.append(contour)
                        total_cell_count += 1

            mask_image = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(mask_image, all_contours, -1, 255, thickness=cv2.FILLED)
            cv2.imshow("Combined Mask", mask_image)
            cv2.imwrite(f"Combined_Mask_{file_name}.jpg", mask_image)

            cv2.putText(original_image, f"Total Cell Count: {total_cell_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Bounding Boxes on Original Image", original_image)
            cv2.imwrite(f"annotated_cells+original_image_{file_name}.jpg", original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(f"Total cells found in {file_name}: {total_cell_count}")

# Example usage
classified_folder = "./cell_dataset/weka"  # Folder with Weka classified images
original_images_folder = "./cell_dataset/orginal"  # Folder with original images
output_folder = "./cell_dataset/weka"  # Folder to save YOLO annotations
convert_and_visualize_on_original(classified_folder, original_images_folder, output_folder)