
import matplotlib.pyplot as plt


def plotting(pre_pross, y_test, y_pred_original_scale):
    print('\n**Defining and plotting the data**\n')

    # Plotting 4 different timeframes of the output data
    # Last 500 Rows of Price
    x_close_500 = pre_pross['close'][-500:]
    x_ema_500 = pre_pross['EMA50'][-500:]
    x_ema_200 = pre_pross['EMA200'][-500:]
    x_ema_800 = pre_pross['EMA800'][-500:]
    # time_500 = pre_pross['time'][-500:]

    # Last 500 Rows of Dataset
    y_test_500 = y_test[-500:]
    y_pred_500 = y_pred_original_scale[-500:]

    # Last 50 Rows of Price
    x_close_50 = pre_pross['close'][-50:]
    x_ema_50 = pre_pross['EMA50'][-50:]
    x_ema_5 = pre_pross['EMA5'][-50:]
    x_ema_13 = pre_pross['EMA13'][-50:]
    # time_50 = pre_pross['time'][-50:]

    # Last 50 Rows of Dataset
    y_test_50 = y_test[-50:]
    y_pred_50 = y_pred_original_scale[-50:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # PRICE LONG
    axes[0, 0].plot(x_close_500, color = 'black', label = 'Close Price')
    axes[0, 0].plot(x_ema_500, color = 'green', label = '50 EMA')
    axes[0, 0].plot(x_ema_200, color = 'red', label = '200 EMA')
    axes[0, 0].plot(x_ema_800, color = 'blue', label = '800 EMA')
    axes[0, 0].legend()
    axes[0, 0].set_title('Last 500 Rows of Price')
    axes[0, 0].grid(True)

    # PREDICTION LONG
    axes[0, 1].plot(y_test_500, color = 'black', label = 'Test')
    axes[0, 1].plot(y_pred_500, color = 'red', label = 'Pred')
    axes[0, 1].legend()
    axes[0, 1].set_title('Last 500 Rows of Dataset')
    axes[0, 1].grid(True)

    # PRICE SHORT
    axes[1, 0].plot(x_close_50, color = 'black', label = 'Close Price')
    axes[1, 0].plot(x_ema_5, color = 'green', label = '5 EMA')
    axes[1, 0].plot(x_ema_13, color = 'red', label = '13 EMA')
    axes[1, 0].plot(x_ema_50, color = 'blue', label = '50 EMA')
    axes[1, 0].legend()
    axes[1, 0].set_title('Last 50 Rows of Price')
    axes[1, 0].grid(True)

    # PREDICTION SHORT
    axes[1, 1].plot(y_test_50, color = 'black', label = 'Test')
    axes[1, 1].plot(y_pred_50, color = 'red', label = 'Pred')
    axes[1, 1].legend()
    axes[1, 1].set_title('Last 50 Rows of Dataset')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()