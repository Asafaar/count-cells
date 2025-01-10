import numpy as np
from sklearn.model_selection import train_test_split

# טוען את התמונה ומפת ההסתברויות
image = "./train/Sample_100.tiff"  # תמונה (לדוגמה: 512x512)
prob_map = "./train/Probability maps.tif"  # מפת הסתברויות (לדוגמה: 512x512)

# סף כדי להפוך את מפת ההסתברויות לתוויות בינאריות
threshold = 0.5
labels = (int(prob_map) > threshold).astype(int)

# הכנת תכונות (Features)
height, width = image.shape
X = []  # רשימה לתכונות
y = []  # רשימה לתוויות
for i in range(height):
    for j in range(width):
        pixel_intensity = image[i, j]
        X.append([pixel_intensity, i, j])  # התכונות: עוצמת פיקסל + מיקום
        y.append(labels[i, j])  # התווית: האם זה תא או לא

X = np.array(X)
y = np.array(y)

# חלוקה לנתוני אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# יצירת המודל
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# חיזוי ובדיקה
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
predicted_labels = clf.predict(X).reshape(image.shape)

# מציאת תאים (ספירת קונטוריות)
import cv2

# הפיכת התוויות הבינאריות למסכה
binary_mask = (predicted_labels == 1).astype(np.uint8)

# מציאת קונטוריות של תאים
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cell_count = len(contours)
print(f"מספר התאים שנמצאו בתמונה: {cell_count}")
