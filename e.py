from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def count_cells(image_path, conf_threshold=0.25):
    """
    ספירת אובייקטים בתמונה באמצעות מודל YOLO ברירת מחדל
    
    Args:
        image_path (str): הנתיב לתמונה
        conf_threshold (float): סף הביטחון המינימלי לזיהוי
    
    Returns:
        tuple: (מספר האובייקטים שזוהו, התמונה המוערת)
    """
    # טעינת מודל ברירת המחדל
    model = YOLO('./yolov8l.pt')  # משתמשים במודל הקטן ביותר לדוגמה
    
    # טעינת התמונה
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("לא ניתן לטעון את התמונה")
    
    # ביצוע הזיהוי
    results = model(image)[0]
    
    # ספירת האובייקטים וציור הזיהויים
    object_count = 0
    annotated_image = image.copy()
    
    # מילון של שמות המחלקות
    class_names = model.names
    
    for r in results.boxes.data:
        x1, y1, x2, y2, score, class_id = r
        class_id = int(class_id)
        if score > conf_threshold:
            object_count += 1
            
            # ציור הריבוע
            cv2.rectangle(
                annotated_image, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )
            
            # הוספת טקסט עם שם המחלקה ורמת הביטחון
            text = f'{class_names[class_id]} {score:.2f}'
            cv2.putText(
                annotated_image, 
                text, 
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
    
    return object_count, annotated_image

# קוד לדוגמה
def main():
    # הורדת מודל ברירת המחדל אם צריך
    model = YOLO('yolov8l.pt')
    
    # נתיב לתמונת הדוגמה שלך
    image_path = "./הורדה.jpg"
    
    try:
        # ספירת אובייקטים בתמונה
        count, annotated = count_cells(image_path)
        
        # הדפסת התוצאות
        print(f"נמצאו {count} אובייקטים בתמונה")
        
        # שמירת התמונה המוערת
        cv2.imwrite("annotated_result.jpg", annotated)
        
        # הצגת התמונה (אופציונלי)
        cv2.imshow('Detected Objects', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"שגיאה: {str(e)}")

if __name__ == "__main__":
    main()