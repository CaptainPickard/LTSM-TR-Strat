## LTSM-TR-Strat

# Python project for creating market LTSM training models

Most of this is just for learning purposes modeling "forward" testing trading strategies. using simple EMA and RSI values. 

The PMP.py File has everything one would need to test and train a LTSM model with tensorflow, specifically designed to predict daily price action returns, and not price itself.

Some key differences with my Tensorflow approach..

1. The model does not use a mim max scaler, why? Becasue min max scaling is terrible for predicting price action. Asset prices have ever incresing all time highs. which throw the model out off since min max scaling is bound between ranges 0 - 1, and cant capture ever increasing all time highs. Thats why this approach use standardization as its scaler function.

2. The model is designed for predicting price returns, not price itself. Predicting the actual closing price of an asset is a fools errand, and why almost all models fail at getting accurate predictions. This model takes a different approach, and trys to predict the daily percentage either up or down.
