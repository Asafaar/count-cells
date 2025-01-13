import cv2
from ultralytics import YOLO
import numpy as np

def detect_and_highlight_empty_areas(image_path, model_path, color=(0, 0, 255)): # color = red
    model = YOLO(model_path)
    results = model(image_path)

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    mask = np.ones((h, w), dtype=np.uint8) * 255  

    for box in results[0].boxes:
        xyxy = box.xyxy[0].int().tolist()
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 0, 0), -1)  

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.fillPoly(img, [contour], color) 

    cv2.imshow("Original Image with Empty Areas Highlighted", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# דוגמת שימוש:
image_path = "./datasets/data/train/images/SIMCEPImages_A03_C10_F1_s09_w2.TIF"
model_path = "./modetrain/runs/detect/train8/weights/best.pt"
detect_and_highlight_empty_areas(image_path, model_path)