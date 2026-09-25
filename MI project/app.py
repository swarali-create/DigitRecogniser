# digit_gui_fixed.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pickle



import sys, os

# Get correct path whether running as .py or .exe
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, 'model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)



root = tk.Tk()
root.title("Digit Recognizer")
root.geometry("400x480")
root.configure(bg="white")

canvas = tk.Canvas(root, width=280, height=280, bg='black', cursor="cross")
canvas.pack(pady=20)

image = Image.new("L", (280, 280), color=0)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=0)
    result_label.config(text="")

def predict():
    # Convert to 28x28
    img_resized = image.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(img_resized).reshape(1, -1)
    # Model was trained on 0-255 grayscale values
    pred = model.predict(img_array)[0]
    result_label.config(text=f"Prediction: {pred}", font=("Arial", 18, "bold"))

canvas.bind("<B1-Motion>", paint)

btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict, bg="#4CAF50", fg="white", width=10).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Clear", command=clear, bg="#f44336", fg="white", width=10).grid(row=0, column=1, padx=10)

result_label = tk.Label(root, text="", bg="white", fg="black")
result_label.pack(pady=10)

root.mainloop()
