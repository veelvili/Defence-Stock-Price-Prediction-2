# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def add_technical_indicators(df):
    """Calculates and adds 7 common technical indicators to the dataframe."""
    df_temp = df.copy()
    df_temp.set_index('Date', inplace=True)

    # Relative Strength Index (RSI)
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df_temp['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_temp['Close'].ewm(span=26, adjust=False).mean()
    df_temp['MACD'] = exp1 - exp2
    df_temp['MACD_signal'] = df_temp['MACD'].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator
    low_14 = df_temp['Low'].rolling(window=14).min()
    high_14 = df_temp['High'].rolling(window=14).max()
    df_temp['Stoch_K'] = ((df_temp['Close'] - low_14) / (high_14 - low_14)) * 100
    df_temp['Stoch_D'] = df_temp['Stoch_K'].rolling(window=3).mean()

    # Average True Range (ATR)
    high_low = df_temp['High'] - df_temp['Low']
    high_prev_close = np.abs(df_temp['High'] - df_temp['Close'].shift())
    low_prev_close = np.abs(df_temp['Low'] - df_temp['Close'].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df_temp['ATR'] = tr.rolling(window=14).mean()

    # On-Balance Volume (OBV)
    df_temp['OBV'] = (np.sign(df_temp['Close'].diff()) * df_temp['Volume']).fillna(0).cumsum()

    df_temp.reset_index(inplace=True)
    return df_temp
