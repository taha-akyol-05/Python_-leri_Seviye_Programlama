import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# ✅ Sınıf isimlerini buraya yaz (eğittiğin sınıflar neyse)
class_names = ['bacterial', 'corona virus', 'normal', 'tuberculosis', 'viral']

# ✅ EfficientNetB0 modelini oluştur
def build_model():
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

# ✅ Modeli oluştur ve ağırlıkları yükle
model = build_model()
model.load_weights("trained_model_weights.h5")

# ✅ Görseli ön işleme (padding dahil)
def preprocess_image(image_path):
    img = load_img(image_path)
    img = img.resize((224, 224))  # padding yapılmadıysa bu basit resize yeterlidir
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # (1, 224, 224, 3)
    return img_array

# ✅ Tahmin fonksiyonu
def predict(image_path):
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[predicted_index], confidence

# ✅ Tkinter arayüzü
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Görseli göster
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

        # Tahmin et ve sonucu göster
        label, confidence = predict(file_path)
        result_label.config(text=f"Tahmin: {label}\nGüven: {confidence:.2f} %")

# ✅ Arayüzü başlat
window = tk.Tk()
window.title("Akciğer Görüntü Sınıflandırıcı")
window.geometry("400x400")

btn = Button(window, text="Görsel Seç", command=select_image)
btn.pack(pady=10)

panel = Label(window)
panel.pack()

result_label = Label(window, text="Tahmin sonucu burada gösterilecek", font=("Arial", 12))
result_label.pack(pady=10)

window.mainloop()
