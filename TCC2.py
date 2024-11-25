import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_curve, auc, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from prophet import Prophet

# Carregar os dados
url = 'https://docs.google.com/spreadsheets/d/1K2_82yMQZVzSS8ISOmysYcGC3dMFl4STIvPIQQjMKgE/export?format=csv'
df = pd.read_csv(url, parse_dates=['ds'])

# Pré-processamento dos dados
df['ds'] = pd.to_datetime(df['ds'], format='%d-%m-%Y', errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
opinion_mapping = {'buy': 1, 'sell': -1, 'hold': 0, 'strong sell': -2}
df['Investing_num'] = df['Investing'].map(opinion_mapping)
df['BTG_num'] = df['BTG'].map(opinion_mapping)
df['XP_num'] = df['XP'].map(opinion_mapping)
df.fillna(0, inplace=True)

# Configuração para validação: Últimos 45 registros como teste
train_set = df.iloc[:-45]
test_set = df.iloc[-45:]

# ===================================================================
# Modelo Prophet
# ===================================================================
print("=== Executando o modelo Prophet ===")

# Criar e ajustar o modelo Prophet
prophet_model = Prophet()
prophet_model.fit(train_set.rename(columns={'ds': 'ds', 'y': 'y'}))

# Fazer previsões
future = test_set[['ds']]
y_pred_prophet = prophet_model.predict(future)

# Avaliar o modelo Prophet
mae_prophet = mean_absolute_error(test_set['y'], y_pred_prophet['yhat'])
mse_prophet = mean_squared_error(test_set['y'], y_pred_prophet['yhat'])
rmse_prophet = np.sqrt(mse_prophet)

print(f"Prophet - MAE: {mae_prophet:.2f}, MSE: {mse_prophet:.2f}, RMSE: {rmse_prophet:.2f}")

# Visualizar resultados do Prophet
plt.figure(figsize=(12, 6))
plt.plot(test_set['ds'], test_set['y'], label='Valores Reais', color='blue')
plt.plot(test_set['ds'], y_pred_prophet['yhat'], label='Prophet', color='orange', linestyle='--')
plt.title('Comparação de Previsões - Prophet')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.show()

# ===================================================================
# Modelo LSTM com análise de opinião
# ===================================================================
print("=== Executando o modelo LSTM ===")

# Normalização
scaler_y = MinMaxScaler()
scaler_features = MinMaxScaler()
df['y_scaled'] = scaler_y.fit_transform(df[['y']])
features_scaled = scaler_features.fit_transform(df[['Investing_num', 'BTG_num', 'XP_num']])
df_scaled = np.hstack((df['y_scaled'].values.reshape(-1, 1), features_scaled))

# Criar janelas para o LSTM (45 dias)
window_size = 45
X, y = [], []
for i in range(window_size, len(df_scaled)):
    X.append(df_scaled[i-window_size:i, :])
    y.append(df_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Dividir em treino e teste
test_size = 45
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# Construção do modelo LSTM
lstm_model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinar o modelo LSTM
history_lstm = lstm_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping_lstm],
    verbose=1
)

# Fazer previsões
y_pred_lstm = lstm_model.predict(X_test)
y_pred_lstm_rescaled = scaler_y.inverse_transform(y_pred_lstm)
y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Avaliar o modelo LSTM
mae_lstm = mean_absolute_error(y_test_rescaled, y_pred_lstm_rescaled)
mse_lstm = mean_squared_error(y_test_rescaled, y_pred_lstm_rescaled)
rmse_lstm = np.sqrt(mse_lstm)

print(f"LSTM - MAE: {mae_lstm:.2f}, MSE: {mse_lstm:.2f}, RMSE: {rmse_lstm:.2f}")

# Visualizar resultados do LSTM
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Valores Reais', color='blue')
plt.plot(y_pred_lstm_rescaled, label='LSTM', color='red', linestyle='--')
plt.title('Previsões LSTM vs Valores Reais')
plt.xlabel('Amostras')
plt.ylabel('Valor')
plt.legend()
plt.show()

# ===================================================================
# Sistema Combinado - Rede Neural
# ===================================================================
print("=== Ajustando a Rede Neural para Combinação ===")

# Ajustar previsões para combinação
aligned_ds = test_set['ds'].values
aligned_y_real = test_set['y'].values
aligned_lstm_pred = y_pred_lstm_rescaled.flatten()
aligned_prophet_pred = y_pred_prophet['yhat'].values[-45:]

# Criar DataFrame com previsões
combined_predictions = pd.DataFrame({
    'ds': aligned_ds,
    'y_real': aligned_y_real,
    'lstm_pred': aligned_lstm_pred,
    'prophet_pred': aligned_prophet_pred
})

# Normalização
scaler_combined = MinMaxScaler()
X_combined = scaler_combined.fit_transform(combined_predictions[['lstm_pred', 'prophet_pred']])
y_combined = scaler_combined.fit_transform(combined_predictions[['y_real']])

# Dividir em treino e teste
split_index = len(y_combined) // 2
X_train_comb, X_test_comb = X_combined[:split_index], X_combined[split_index:]
y_train_comb, y_test_comb = y_combined[:split_index], y_combined[split_index:]

# Construção do modelo combinado
combined_model = Sequential([
    Dense(64, activation='relu', input_dim=2),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

combined_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stopping_comb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinamento do modelo combinado
combined_model.fit(
    X_train_comb, y_train_comb,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping_comb]
)

# Fazer previsões com o modelo combinado
y_pred_comb_scaled = combined_model.predict(X_test_comb)
y_pred_comb_rescaled = scaler_combined.inverse_transform(y_pred_comb_scaled)

# Avaliar o modelo combinado
mae_comb = mean_absolute_error(combined_predictions['y_real'][split_index:], y_pred_comb_rescaled.flatten())
mse_comb = mean_squared_error(combined_predictions['y_real'][split_index:], y_pred_comb_rescaled.flatten())
rmse_comb = np.sqrt(mse_comb)

print(f"Combinado - MAE: {mae_comb:.2f}, MSE: {mse_comb:.2f}, RMSE: {rmse_comb:.2f}")

# Visualizar resultados do modelo combinado
plt.figure(figsize=(12, 6))
plt.plot(combined_predictions['ds'][split_index:], combined_predictions['y_real'][split_index:], label='Valores Reais', color='blue')
plt.plot(combined_predictions['ds'][split_index:], y_pred_comb_rescaled.flatten(), label='Combinado', color='green', linestyle='--')
plt.title('Previsões Combinadas')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.show()
# ===================================================================
# Gráficos Comparativos
# ===================================================================
plt.figure(figsize=(12, 6))
plt.plot(combined_predictions['ds'], combined_predictions['y_real'], label='Valores Reais', color='blue')
plt.plot(combined_predictions['ds'], combined_predictions['prophet_pred'], label='Prophet', color='orange', linestyle='--')
plt.plot(combined_predictions['ds'], combined_predictions['lstm_pred'], label='LSTM', color='red', linestyle='--')
plt.plot(combined_predictions['ds'][split_index:], y_pred_comb_rescaled.flatten(), label='Combinado', color='green', linestyle='--')
plt.title('Comparação de Previsões - Modelos')
plt.xlabel('Data')
plt.ylabel('Valor')
plt.legend()
plt.show()

# ===================================================================
# Curva ROC e Acurácia - Modelo Combinado
# ===================================================================
comb_real_binary = (combined_predictions['y_real'][split_index:] > np.median(combined_predictions['y_real'][split_index:])).astype(int)
comb_pred_binary = (y_pred_comb_rescaled.flatten() > np.median(y_pred_comb_rescaled.flatten())).astype(int)

fpr_comb, tpr_comb, _ = roc_curve(comb_real_binary, y_pred_comb_rescaled.flatten())
roc_auc_comb = auc(fpr_comb, tpr_comb)
accuracy_comb = accuracy_score(comb_real_binary, comb_pred_binary)

print(f"Combinado - AUC: {roc_auc_comb:.2f}, Acurácia: {accuracy_comb:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_comb, tpr_comb, label=f'Combinado (AUC = {roc_auc_comb:.2f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Curva ROC - Modelo Combinado')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend()
plt.show()



