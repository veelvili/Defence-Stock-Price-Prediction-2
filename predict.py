# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
from technical_indicators import add_technical_indicators

def load_hybrid_model(model_dir):
    """Load all components of the hybrid model."""
    # Load models
    lstm_model = load_model(os.path.join(model_dir, 'lstm_model.h5'))
    svr_model = joblib.load(os.path.join(model_dir, 'svr_model_tuned.pkl'))
    meta_model = joblib.load(os.path.join(model_dir, 'meta_model.pkl'))
    
    # Load scalers
    scaler_features = joblib.load(os.path.join(model_dir, 'scaler_features.pkl'))
    scaler_target = joblib.load(os.path.join(model_dir, 'scaler_target.pkl'))
    
    # Load configuration
    config = joblib.load(os.path.join(model_dir, 'model_config.pkl'))
    
    return lstm_model, svr_model, meta_model, scaler_features, scaler_target, config

def predict_stock_price(model_dir, ticker, start_date, end_date, geo_data_path):
    """Predict stock prices using the hybrid model."""
    # Load all model components
    lstm_model, svr_model, meta_model, scaler_features, scaler_target, config = load_hybrid_model(model_dir)
    
    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    stock_data = stock_data.reset_index()
    stock_data.columns = stock_data.columns.get_level_values(0)
    stock_data.dropna(inplace=True)
    
    # Add technical indicators
    stock_data = add_technical_indicators(stock_data)
    
    # Load and merge geopolitical data
    geo_df = pd.read_csv(geo_data_path)
    geo_df['Date'] = pd.to_datetime(geo_df['Date'], dayfirst=True)
    geo_df.set_index('Date', inplace=True)
    geo_df.index.name = 'Date'
    
    stock_data.set_index('Date', inplace=True)
    stock_data_full = stock_data.join(geo_df, on='Date', how='left')
    stock_data_full.fillna(method='ffill', inplace=True)
    stock_data_full.dropna(inplace=True)
    
    # Select features
    data_features = stock_data_full[config['feature_columns']]
    
    # Scale features
    scaled_features = scaler_features.transform(data_features)
    
    # Prepare data for LSTM
    time_step = config['time_step']
    X_lstm = []
    for i in range(len(scaled_features) - time_step):
        X_lstm.append(scaled_features[i:(i + time_step)])
    X_lstm = np.array(X_lstm)
    
    # Prepare data for SVR
    X_svr = scaled_features[time_step:]
    
    # Make predictions
    lstm_pred = lstm_model.predict(X_lstm)
    svr_pred = svr_model.predict(X_svr)
    
    # Inverse transform predictions
    lstm_pred_inv = scaler_target.inverse_transform(lstm_pred)
    svr_pred_inv = scaler_target.inverse_transform(svr_pred.reshape(-1, 1))
    
    # Create meta features
    X_meta = pd.DataFrame({
        'LSTM_Pred': lstm_pred_inv.flatten(),
        'SVR_Pred': svr_pred_inv.flatten()
    })
    
    # Make final prediction with meta model
    hybrid_pred = meta_model.predict(X_meta)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': stock_data_full.index[time_step:],
        'Predicted_Close': hybrid_pred
    })
    
    return results

if __name__ == "__main__":
    # Example usage
    model_dir = 'saved_models'
    ticker = 'LMT'
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    geo_data_path = 'geopolitical_data.csv'
    
    predictions = predict_stock_price(model_dir, ticker, start_date, end_date, geo_data_path)
    print(predictions.head())
