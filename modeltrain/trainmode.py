from ultralytics import YOLO
import cv2
import os


data_yaml_path = 'data.yaml'
test_image_path = "./data/SIMCEPImages_A03_C10_F1_s09_w2.jpg"
trained_weights_path = 'runs/detect/train/best.pt' 


if os.path.exists(trained_weights_path):
    model = YOLO(trained_weights_path)
    print("Loading existing model")
else:
    model = YOLO('yolov8n.pt')  
    print("Training new model")
    results = model.train(data=data_yaml_path, epochs=150, imgsz=696) # הגדלת כמות האפוקות


model = YOLO(trained_weights_path) # Load trained model

# val of the model
metrics = model.val()

# Export the model to ONNX format
model.export(format="onnx")

# Count cells in an image
def count_cells(image_path, model):
    results = model(image_path)
    num_cells = len(results[0].boxes)
    print(results[0].boxes)
    print(f"Number of boxes detected: {num_cells}")
    return num_cells


num_cells = count_cells(test_image_path, model)
print(f"Number of cells in {test_image_path}: {num_cells}")

results = model(test_image_path)
annotated_frame = results[0].plot()
cv2.imshow("YOLO Results", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()