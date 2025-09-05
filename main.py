# ===============================
# 1. Import library yang dibutuhkan
# ===============================
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ===============================
# 2. Load dataset CSV
# ===============================
df = pd.read_csv("mycosense_full_dataset.csv")
print(df.head())

# ===============================
# 3. Pisahkan fitur (X) dan label (y)
# ===============================
# Fitur input kita: raw_mV dan amplified_V (sinyal tegangan)
X = df[["raw_mV", "amplified_V"]].values

# Label output: pollutant_label
y = df["pollutant_label"].values

# ===============================
# 4. Encode label (string -> angka)
# ===============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Simpan mapping label untuk interpretasi nanti
print("Mapping Label:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

# ===============================
# 5. Normalisasi fitur
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 6. Split data jadi train & test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===============================
# 7. Bangun model neural network sederhana
# ===============================
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # layer input
    Dropout(0.2),                                                  # regularisasi
    Dense(16, activation='relu'),                                  # hidden layer
    Dropout(0.2),
    Dense(len(encoder.classes_), activation='softmax')             # output (klasifikasi multi-class)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ===============================
# 8. Latih model
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    verbose=1
)

# ===============================
# 9. Evaluasi model
# ===============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi di data test: {acc*100:.2f}%")

# ===============================
# 10. Contoh prediksi baru
# ===============================
# misalnya sinyal tegangan = 0.82 Volt
sample = np.array([[0.82, 0.82]])
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
predicted_class = encoder.inverse_transform([np.argmax(pred)])

print("Hasil prediksi untuk sampel 0.82 V:", predicted_class[0])
