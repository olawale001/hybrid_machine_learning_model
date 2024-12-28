import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


data = pd.read_csv('apple_stock_data.csv')
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]
print(data.columns)

scaler = MinMaxScaler()
data['Scaled_Close'] = scaler.fit_transform(data[['Close']])


def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x, y = create_sequences(data['Scaled_Close'].values, seq_length)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(x_train, y_train, epochs=20, batch_size=32)


data['lag_1'] = data['Scaled_Close'].shift(1)
data['lag_2'] = data['Scaled_Close'].shift(2)
data['lag_3'] = data['Scaled_Close'].shift(3)
data = data.dropna()

x_lin = data[['lag_1', 'lag_2', 'lag_3']]
y_lin = data['Scaled_Close']
x_train_lin, x_test_lin = x_lin[:train_size], x_lin[train_size:]
y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]


lin_model = LinearRegression()
lin_model.fit(x_train_lin, y_train_lin)


x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
lstm_predictions = lstm_model.predict(x_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)


lin_test_start = len(data) - len(x_test_lin) 
lin_test_end = lin_test_start + len(x_test)  
x_test_lin = x_lin.iloc[lin_test_start:lin_test_end]
lin_predictions = lin_model.predict(x_test_lin)
lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))


hybrid_predictions = (0.7 * lstm_predictions) + (0.3 * lin_predictions)


lstm_future_predictions = []
last_sequence = x[-1].reshape(1, seq_length, 1)
for _ in range(15):
    lstm_pred = lstm_model.predict(last_sequence)[0, 0]
    lstm_future_predictions.append(lstm_pred)
    lstm_predict_reshape = np.array([[lstm_pred]]).reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], lstm_predict_reshape, axis=1)
lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))

recent_data = data['Scaled_Close'].values[-3:]
lin_future_predictions = []
for _ in range(15):
    lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
    lin_future_predictions.append(lin_pred)
    recent_data = np.append(recent_data[1:], lin_pred)
lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))

hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)


future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=15)
prediction_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM Prediction': lstm_future_predictions.flatten(),
    'Linear Regression Predictions': lin_future_predictions.flatten(),
    'Hybrid Model Predictions': hybrid_future_predictions.flatten()
})

print(prediction_df)
