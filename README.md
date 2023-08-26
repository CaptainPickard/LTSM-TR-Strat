# LTSM-TR-Strat

## Python project for creating market LTSM training models

Most of this is just for learning purposes, an attempt at modeling "forward" testing trading strategies, using EMA and RSI values. 

This project works for both Crypto and FOREX. The inputs required to run are done via the command line.

The main.py File has everything one would need to test and train a LTSM model with tensorflow, specifically designed to predict daily price action returns, and not price itself.

Some key differences with my Tensorflow approach..

1. The model does not use a mim max scaler. Asset prices have uncapped highs, which will throw off training models since min max scaling is bound between ranges 0 - 1, and cant capture ever increasing ATHs. Thats why this approach use standardization as its scaler function.

2. The model is designed for predicting price returns, not price itself. Predicting the actual closing price of an asset is a fools errand, and why almost all models fail at getting accurate predictions. This model takes a different approach, and trys to predict the daily percentage either up or down.

### To Get Started

Install the following dependencies
- Tensorflow
- pandas
- matplotlib
- sklearn
- numpy
- keras

Once dependencies are installed in an env, you're ready to run the main.py.

Alos, there is a save function to this as well. you can choose to save the output in csv. My orignal idea was to use these outputs to train future models, but I never got around to that part.

### Future Features

1. Adding in federal reserve data to use when training the model, data such as...
- CPI (Consumer Price Index)
- NON-FARM data
- Interest rate Increases/Decreases
- ETC

Price moves according to the news, if human behavior can be modeled in terms of its reaction to positive or negative news, the model could be quite valuable.
