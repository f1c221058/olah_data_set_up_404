
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Baca data
data = pd.read_csv("dat_penelitian.csv", delimiter=';')

# Ambil fitur dan target
X = data[['suhu ', 'kelembapan (%)', 'tekanan(kPa)', 'kecepatang angin(m/s)']]
y = data['curah hujan(mm)']

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Bangun model JST
model = Sequential()
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Kompilasi dan pelatihan
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Jika ingin mengembalikan hasil ke bentuk asli
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

import matplotlib.pyplot as plt
import numpy as np

# Invers transform hasil prediksi dan data uji
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# Plot grafik
plt.figure(figsize=(14, 8))
plt.plot(y_test_inv, 'ro-', label='Data Aktual')
plt.plot(y_pred_inv, 'b^-', label='Prediksi JST')
plt.title('Perbandingan Curah Hujan Aktual vs Prediksi JST')
plt.xlabel('Indeks Data')
plt.ylabel('Curah Hujan (mm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
