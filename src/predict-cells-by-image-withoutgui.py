import cv2
from ultralytics import YOLO


model = YOLO('./modetrain/runs/detect/train8/weights/best.pt')  # נתיב למודל שלך


results = model('./datasets/data/train/images/SIMCEPImages_A03_C10_F1_s09_w2.TIF')  # נתיב לתמונה שלך


for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    cv2.imshow('image', im_array)
    cv2.waitKey(0)
    cv2.imwrite('runs/detect/predictions.jpg', im_array[..., ::-1])

cv2.destroyAllWindows()