from ultralytics import YOLO
import cv2
import os

# נתיבים
data_yaml_path = 'data.yaml'
test_image_path = "./data/SIMCEPImages_A03_C10_F1_s09_w2.jpg"
trained_weights_path = 'runs/detect/train/best.pt' # נתיב משתנה לאחר כל אימון

# טעינת או אימון המודל
if os.path.exists(trained_weights_path):
    model = YOLO(trained_weights_path)
    print("Loading existing model")
else:
    model = YOLO('yolov8n.pt')  # טעינת מודל מאומן מראש (חשוב!)
    print("Training new model")
    results = model.train(data=data_yaml_path, epochs=150, imgsz=696) # הגדלת כמות האפוקות

# וידוא שהאימון הסתיים בהצלחה לפני המשך
# if not os.path.exists(trained_weights_path):
#     print("Training failed or did not complete. Exiting.")
#     exit()

model = YOLO(trained_weights_path) # טעינה מחדש של המודל עם המשקלים המאומנים

# הערכת ביצועים (אופציונלי, מומלץ לאחר האימון)
metrics = model.val()

# שמירת המודל המאומן (אופציונלי, נשמר אוטומטית)
model.export(format="onnx")

# פונקציה לספירה (ללא שינוי משמעותי)
def count_cells(image_path, model):
    results = model(image_path)
    num_cells = len(results[0].boxes)
    print(results[0].boxes)
    print(f"Number of boxes detected: {num_cells}")
    return num_cells

# שימוש במודל לספירה והצגה
num_cells = count_cells(test_image_path, model)
print(f"Number of cells in {test_image_path}: {num_cells}")

results = model(test_image_path)
annotated_frame = results[0].plot()
cv2.imshow("YOLO Results", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()