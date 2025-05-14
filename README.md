İleri Python Proje
EfficientNetB0 ile Akciğer Hastalıkları Sınıflandırma Sistemi
Veri seti: [Kaggle - Chest X-ray Dataset](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types/data)

1. Projenin Genel Amacı
Bu projede, EfficientNetB0 tabanlı bir derin öğrenme modeli kullanılarak, akciğer röntgen görüntülerinden bacterial pneumonia, corona virus (COVID-19), normal, tuberculosis ve viral pneumonia olmak üzere beş farklı hastalığın sınıflandırılması amaçlanmıştır.

Modelin öğrenmesi ve genelleme başarısı, veri artırma teknikleri, sınıf dengesizliklerini azaltan stratejiler ve optimizasyon yöntemleriyle güçlendirilmiştir. Projenin temel hedefi, sağlık alanında doğru ve hızlı tanı desteği sunabilecek bir yapay zeka sistemi geliştirmektir.

2. Kullanılan Teknolojiler ve Kütüphaneler
Python: Projenin temel programlama dili.

TensorFlow & Keras: Derin öğrenme modeli oluşturma ve eğitme.

Pandas & NumPy: Veri işleme ve düzenleme işlemleri için.

Matplotlib & Seaborn: Görselleştirme (accuracy, loss grafikleri, karışıklık matrisi).

Scikit-learn: Sınıflandırma raporu ve F1-score hesaplamaları için.

OpenCV / PIL: Görüntü işleme işlemleri.

Imbalanced-learn: Sınıf ağırlıkları ve örnekleme stratejileri.

Google Colab / Jupyter Notebook: Geliştirme ortamı.

OS & glob: Dosya işlemleri için.

3. Kodun Ana Bölümleri
a. Veri Ön İşleme
Görüntüler yeniden boyutlandırıldı (224x224).

Normalize edildi (0-1 aralığına).

Etiketler, one-hot encoding yöntemiyle dönüştürüldü.

b. Model Mimarisi: EfficientNetB0
ImageNet ağırlıklarıyla önceden eğitilmiş EfficientNetB0 modeli kullanıldı.

Üst katmanlar özelleştirildi: GlobalAveragePooling, Dense, Dropout ve Softmax katmanları eklendi.

c. Eğitim Stratejisi
Kayıp fonksiyonu: Categorical Crossentropy.

Optimizasyon: SGD (low learning rate), momentum.

Regularizasyon: Dropout ve Gradient Clipping uygulandı.

EarlyStopping & ModelCheckpoint callback’leri ile eğitim optimize edildi.

d. Performans Ölçümü
Accuracy, precision, recall, F1-score gibi metrikler hesaplandı.

Karışıklık matrisi ile modelin sınıflar arası başarı oranları analiz edildi.

e. Veri Artırma ve Sınıf Dengesi
ImageDataGenerator ile döndürme, yakınlaştırma, yansıma gibi veri artırma işlemleri uygulandı.

Class weight hesaplandı ve dengesiz sınıflar için eğitim sırasında ağırlıklar kullanıldı.

Projenin İşleyişi
Görüntü Verisi ile Eğitimi Başlat
Kullanıcı, etiketlenmiş veri setini sağlar.

Model, görüntüler üzerinde eğitim sürecini başlatır.

Yeni Görüntü Sınıflandırma
Kullanıcı, yeni bir röntgen görüntüsü yüklediğinde model bunu sınıflandırarak hangi hastalığa ait olduğunu tahmin eder.

Sonuçların Görselleştirilmesi
Eğitim ve doğrulama kayıpları ile doğruluk grafikleri.

Karışıklık matrisi ve sınıf bazlı başarı metrikleri görsellenir.

5. Sonuç
Bu proje, veriye dayalı bir derin öğrenme yaklaşımı ile akciğer hastalıklarının otomatik olarak sınıflandırılmasını hedeflemiştir. EfficientNetB0 mimarisi sayesinde düşük donanımda bile yüksek doğrulukta sonuçlar elde edilmiştir.

En yüksek başarı tuberculosis sınıfında gözlenmiştir. Benzer görsel özelliklere sahip viral ve bacterial pneumonia sınıflarında karışıklık yaşanmış, bu durum ileride daha fazla örnekle ve farklı mimarilerle iyileştirilebilir.

Bu proje, tıbbi görüntüleme alanında kullanılabilecek etkili ve optimize edilmiş bir karar destek sistemi olarak değerlendirilebilir.
