import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

# TensorFlow'un belleği dinamik olarak tahsis etmesini sağla
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Veri seti yolu 
train_dir = 'veri_seti/train'
test_dir = 'veri_seti/test'

# Görselleri padding ile işleme fonksiyonu (224x224 için)
def resize_with_padding(image, target_size=(224, 224)):
    height, width, _ = image.shape
    target_height, target_width = target_size

    # En boy oranını koru
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Görseli yeniden boyutlandır
    image = tf.image.resize(image, (new_height, new_width))

    # Padding ekle
    padded_img = tf.image.pad_to_bounding_box(
        image,
        (target_height - new_height) // 2,
        (target_width - new_width) // 2,
        target_height,
        target_width
    )

    return padded_img

# ImageDataGenerator için özel bir ön işleme fonksiyonu
def custom_preprocessing(image):
    image = resize_with_padding(image, target_size=(224, 224))
    return image

# Veri artırma
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    channel_shift_range=0.3,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,  # Batch size artırıldı
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,  # Batch size artırıldı
    class_mode='categorical',
    shuffle=False
)

# Sınıf dengesizliğini kontrol etme ve class weights hesaplama
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Modeli oluşturma (EfficientNet-B0)
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tuning: Daha fazla katmanı eğitime aç
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Ekstra katmanlar ve Dropout ekleme
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Optimizer (SGD + momentum)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)

# Modeli derleme
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks ekleme
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_weights.weights.h5',  # Colab için uygun dosya yolu
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    mode='max'
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weights_dict
)

# Eğitilmiş en iyi ağırlıkları yükleme
weights_path = 'best_model_weights.weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print(f"Ağırlık dosyası bulunamadı: {weights_path}")

# Yeni tahmin fonksiyonu
def predict_image(image_path):
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = resize_with_padding(img_array, target_size=(224, 224))
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = list(train_generator.class_indices.keys())[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    print(f"Tahmin edilen sınıf: {predicted_class}\nGüven: {confidence:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score

# Modelin tamamını .h5 uzantılı olarak kaydet
model.save_weights("trained_model_weights.h5")


# Tahminler (val seti için)
y_pred_probs = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = validation_generator.classes

# Accuracy, Precision, Recall, F1-score, MCC
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

# Sınıf isimlerini al
class_names = list(train_generator.class_indices.keys())

# Sonuçları yazdır
print(f"\n--- Model Performans Değerleri ---")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"MCC          : {mcc:.4f}")

# Sınıf bazlı detaylı sınıflandırma raporu
print("\n--- Sınıf Bazlı Sınıflandırma Raporu ---")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Karışıklık Matrisi çizimi (opsiyonel)
import seaborn as sns
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()



# Eğitim grafikleri
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test doğruluğu: {test_accuracy * 100:.2f}%")

# Yanlış tahmin edilen örnekleri sınıf sınıf görselleştir
for true_idx in range(len(class_names)):
    for pred_idx in range(len(class_names)):
        if true_idx == pred_idx:
            continue

        confused = [i for i in range(len(y_true)) if y_true[i] == true_idx and y_pred[i] == pred_idx]
        if not confused:
            continue

        print(f"\nGerçek: {class_names[true_idx]} → Tahmin: {class_names[pred_idx]} ({len(confused)} hata)")

        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(confused[:5]):
            img_path = validation_generator.filepaths[idx]
            img = load_img(img_path, target_size=(224, 224))
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{class_names[true_idx]} → {class_names[pred_idx]}")
        plt.tight_layout()
        plt.show()


plt.tight_layout()
plt.show()