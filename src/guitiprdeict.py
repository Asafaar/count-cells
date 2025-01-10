import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from ultralytics import YOLO
import os
import numpy as np
from ttkthemes import ThemedTk

def run_prediction():
    image_path = image_entry.get()
    model_path = model_entry.get()
    scale = scale_combobox.get()
    flask_size = flask_size_combobox.get()

    if not all([image_path, model_path, scale, flask_size]):
        messagebox.showerror("Error", "Please fill all fields.")
        return

    try:
        scale = float(scale)
        flask_size = float(flask_size)
    except ValueError:
        messagebox.showerror("Error", "Invalid scale or flask size. Please select from the dropdown.")
        return

    if not os.path.exists(image_path):
        messagebox.showerror("Error", "Image file not found.")
        return

    if not os.path.exists(model_path):
        messagebox.showerror("Error", "Model file not found.")
        return

    try:
        model = YOLO(model_path)
        results = model(image_path)
        num_cells = len(results[0].boxes)

        im_array = results[0].plot()

        cv2.putText(im_array, f"Cells: {num_cells}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        total_cells_in_flask = num_cells * (flask_size / scale)
        result_label.config(text=f"Cells in image: {num_cells}\nEstimated cells in flask: {int(total_cells_in_flask)}")

        cv2.imshow('Prediction', im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def browse_image():
    filename = filedialog.askopenfilename(initialdir="./", title="Select Image", filetypes=(("Image files", "*.jpg;*.png;*.tif;*.TIF"), ("All files", "*.*")))
    image_entry.delete(0, tk.END)
    image_entry.insert(0, filename)

def browse_model():
    filename = filedialog.askopenfilename(initialdir="./", title="Select Model", filetypes=(("Model files", "*.pt"), ("All files", "*.*")))
    model_entry.delete(0, tk.END)
    model_entry.insert(0, filename)

window = ThemedTk(theme="arc")  # שימוש בערכת נושא "arc". נסה גם "clam", "radiance" וכו'.
window.title("YOLO Cell Counter")

style = ttk.Style(window)
style.configure("TLabel", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12))
style.configure("TCombobox", font=("Arial", 12))
style.configure("TEntry", font=("Arial", 12))

image_label = ttk.Label(window, text="Image Path:")
image_label.grid(row=0, column=0, padx=5, pady=5)

image_entry = ttk.Entry(window, width=50)
image_entry.grid(row=0, column=1, padx=5, pady=5)

image_button = ttk.Button(window, text="Browse", command=browse_image)
image_button.grid(row=0, column=2, padx=5, pady=5)

model_label = ttk.Label(window, text="Model Path:")
model_label.grid(row=1, column=0, padx=5, pady=5)

model_entry = ttk.Entry(window, width=50)
model_entry.grid(row=1, column=1, padx=5, pady=5)

model_button = ttk.Button(window, text="Browse", command=browse_model)
model_button.grid(row=1, column=2, padx=5, pady=5)

scale_label = ttk.Label(window, text="Image Scale (e.g., area in mm^2):")
scale_label.grid(row=2, column=0, padx=5, pady=5)

scale_options = [1, 4, 10, 25, 100]  # דוגמאות לקני מידה
scale_combobox = ttk.Combobox(window, values=scale_options)
scale_combobox.grid(row=2, column=1, padx=5, pady=5)
scale_combobox.current(0)

flask_size_label = ttk.Label(window, text="Flask Size (e.g., area in mm^2):")
flask_size_label.grid(row=3, column=0, padx=5, pady=5)

flask_size_options = [25, 75, 175, 225, 600]  # דוגמאות לגדלי פלאסק
flask_size_combobox = ttk.Combobox(window, values=flask_size_options)
flask_size_combobox.grid(row=3, column=1, padx=5, pady=5)
flask_size_combobox.current(0)

run_button = ttk.Button(window, text="Run Prediction", command=run_prediction)
run_button.grid(row=4, column=1, padx=5, pady=10)

result_label = ttk.Label(window, text="")
result_label.grid(row=5, column=1, padx=5, pady=5)

window.mainloop()