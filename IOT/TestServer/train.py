import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import auto_arima
import joblib

X = pd.read_csv('data.csv')
y = pd.read_csv('label.csv')

def preprocessing(df, task):
    if task == 'Regression':
        Y = y['area'].values
    elif task == 'Classification':
        Y = y['area'].apply(lambda x: 1 if x > 0 else 0).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.60, shuffle=True, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = preprocessing(X, task='Regression')

nn_regressor_model = MLPRegressor(activation='relu', hidden_layer_sizes=(16, 16), max_iter=100, solver='adam')
nn_regressor_model.fit(X_train, Y_train)

model = joblib.dump(nn_regressor_model, 'forestfiremodel.pkl')

dataImport = pd.read_csv('dataImport.csv')

dataImport = dataImport[['FFMC', 'DMC', 'DC', 'ISI', 'temperature', 'humidity', 'wind', 'rain']]

input_data = np.array([])
for i in range(1, 20):
    FFMC_value = dataImport.iloc[i]['FFMC']
    DMC_value = dataImport.iloc[i]['DMC']
    DC_value = dataImport.iloc[i]['DC']
    ISI_value = dataImport.iloc[i]['ISI']
    temperature_value = dataImport.iloc[i]['temperature']
    humidity_value = dataImport.iloc[i]['humidity']
    wind_value = dataImport.iloc[i]['wind']
    rain_value = dataImport.iloc[i]['rain']
    input_data = pd.DataFrame({
        'FFMC': [FFMC_value],
        'DMC': [DMC_value],
        'DC': [DC_value],
        'ISI': [ISI_value],
        'temperature': [temperature_value],
        'humidity': [humidity_value],
        'wind': [wind_value],
        'rain': [rain_value]
    })

    scaler = StandardScaler()
    scaler.fit(X_train)  # X_train là tập dữ liệu huấn luyện đã được chuẩn hóa
    input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # Đưa ra dự đoán
    prediction = nn_regressor_model.predict(input_data_scaled)
    my_array = np.append(input_data, prediction[0])

# Khởi tạo mô hình ARIMA tự động
model = auto_arima(my_array, start_p=1, start_q=0, max_p=2, max_q=2, seasonal=False)

# Dự đoán phần tử thứ 25
next_prediction = model.predict(n_periods=1)

print(next_prediction)