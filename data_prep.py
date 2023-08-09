import pandas as pd
import pandas_ta as ta


def format_data(pre_pross):
    pre_pross['percentage_return'] = pre_pross['close'].pct_change() * 100

    # Processing the data file using some TR ema indicators
    pre_pross['EMA5'] = ta.ema(pre_pross.close, length=5)
    pre_pross['EMA13'] = ta.ema(pre_pross.close, length=13)
    pre_pross['EMA50'] = ta.ema(pre_pross.close, length=50)
    pre_pross['EMA200'] = ta.ema(pre_pross.close, length=200)
    pre_pross['EMA800'] = ta.ema(pre_pross.close, length=800)
    pre_pross['RSI'] = ta.rsi(pre_pross.close, length=15)

    # pre_pross['Target'] = pre_pross['Adj Close']-pre_pross.Open
    # pre_pross['Target'] = pre_pross['Adj Close'].shift(-1)
    # pre_pross['TargetClass'] = [1 if pre_pross.Target[i]>0 else 0 for i in range(len(data))]

    pre_pross['TargetNextClose'] = pre_pross['percentage_return'].shift(-1)
    # pre_pross['TargetNextClose'] = pre_pross['Adj Close'].shift(-1)
    pre_pross.dropna(inplace=True)
    pre_pross.reset_index(inplace=True)
    pre_pross.drop(['volume', 'time'], axis=1, inplace=True)
    print('\n**Formatting Data Complete**\n')
    return pre_pross