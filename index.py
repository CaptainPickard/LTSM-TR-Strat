from historic_data import *
from data_prep import *
from PMP import mlm_model
from plotting_cry import plotting
from data_save import save_to_csv

if __name__ == '__main__':

    while True:
        
        get_asset_input = input("\nChoose: [F] Forex, or [C] Crypto: ").upper()
        
        if get_asset_input == 'C':
            post_pross, ticker = get_user_crypto()
            x_pross = format_crypto_data(post_pross)
            plot1, plot2, plot3 = mlm_model(x_pross)
            
            
        elif get_asset_input == 'F':
            post_pross, ticker = get_user_forex()
            x_pross = format_forex_data(post_pross)
            plot1, plot2, plot3 = mlm_model(x_pross)
        
        print(f"\n Asset: {ticker}\n")

        last_element1 = (plot3[-1])
        print(f'\n Tomorrows {ticker} Predicted Gain/loss: {last_element1}%\n')
        
        test_element2 = (plot2[-1])
        print(f'\n Todays {ticker} Actual Gain/loss: {test_element2}%\n')

        plotting(plot1, plot2, plot3)
        
        user_save = input("\nSave this Prediction (Y/N): ").upper()
        if user_save == 'N':
            pass
        elif user_save == 'Y':
            save_to_csv(last_element1, test_element2, ticker)
        
        user_end = input("\nTry another Asset? (Y/N): ").upper()
        if user_end == 'N':
            break
        elif user_end == 'Y':
            continue
        
    