import tkinter as tk
from tkinter import Canvas, Label, Button
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import windnd
import os

binary_model = load_model('C:/Users/makis/Desktop/Cross-Species Transfer Learning From Dog breed to Cat breed Classification using Deep Learning Techniques/Demo/binary_model.h5')
dogs_model = load_model('C:/Users/makis/Desktop/Cross-Species Transfer Learning From Dog breed to Cat breed Classification using Deep Learning Techniques/Demo/dogs_model.h5')
cats_model = load_model('C:/Users/makis/Desktop/Cross-Species Transfer Learning From Dog breed to Cat breed Classification using Deep Learning Techniques/Demo/cats_model.h5')

def on_drop(files):
    global file_path
    file_path = files[0].decode("utf-8")
    image = Image.open(file_path)
    image = image.resize((640, 480), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo
    canvas.config(width=photo.width(), height=photo.height())


dogs_labels = ['american_pit_bull_terrier', 'beagle', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'japanese_chin', 'leonberger', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'wheaten_terrier', 'yorkshire_terrier']
cats_labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']

root = tk.Tk()
canvas = tk.Canvas(root)
canvas.pack()

label = tk.Label(root, text="Drop picture here", width=30, height=5, relief="solid")
label.pack()

windnd.hook_dropfiles(label, func=on_drop)

def on_predict():
    global file_path
    global binary_model
    global dogs_model
    global cats_model
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = binary_model.predict(x)
    predicted_class = np.argmax(prediction[0])
    if predicted_class == 0:
        prediction2 = cats_model.predict(x)
        predicted_class2 = np.argmax(prediction2[0])
        breed = cats_labels[predicted_class2]
    else:
        prediction2 = dogs_model.predict(x)
        predicted_class2 = np.argmax(prediction2[0])
        breed = dogs_labels[predicted_class2]

    label.config(text=f"Prediction: {breed}")

label = tk.Label(root, text="Prediction: ")
label.pack()

button = tk.Button(root, text="Make Prediction", command=on_predict)
button.pack()

root.mainloop()