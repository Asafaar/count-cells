import cv2

def visualize_yolo_annotations(image_path, annotations_path, class_names):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    cell_count = 0  # Initialize cell counter

    try:
        with open(annotations_path, 'r') as f:
            for line in f:
                cell_count += 1  # Increment counter for each line/cell
                data = line.strip().split()
                class_id = int(data[0])
                x, y, width, height = map(float, data[1:])

                x_center = int(x * w)
                y_center = int(y * h)
                box_width = int(width * w)
                box_height = int(height * h)

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
                cv2.putText(img, "adsf", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) # Add class name

    except FileNotFoundError:
        print(f"Error: Annotations file not found at: {annotations_path}")
        return  # Exit the function if file not found
    except Exception as e: #לכידת שגיאות אחרות
        print(f"An error occurred: {e}")
        return

    cv2.putText(img, f"Cell Count: {cell_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display cell count on image
    cv2.imshow("YOLO Annotations", img)
    cv2.imwrite("YOLO Annotations check.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Total cells found: {cell_count}") # Print cell count to console


# Example usage
image_path = "./cell_dataset/orginal/SIMCEPImages_A03_C10_F1_s09_w2.TIF"
annotations_path = "./cell_dataset/weka/SIMCEPImages_A03_C10_F1_s09_w2.txt"
class_names = ["class1", "class2", "class3"]  # Your class names

visualize_yolo_annotations(image_path, annotations_path, class_names)