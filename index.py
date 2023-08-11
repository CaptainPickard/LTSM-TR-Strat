from historic_data import get_user_crypto
from data_prep import format_data
from PMP import mlm_model
from plotting import plotting
from data_save import save_to_csv

if __name__ == '__main__':

    while True:
        post_pross, ticker = get_user_crypto()
        x_pross = format_data(post_pross)
        plot1, plot2, plot3 = mlm_model(x_pross)
        
        print(f"\n Asset: {ticker}\n")

        last_element1 = (plot3[-1])
        print(f'\n Todays {ticker} Predicted Gain/loss: {last_element1}%\n')
        
        test_element2 = (plot2[-1])
        print(f'\n Tmorrows {ticker} Actual Gain/loss: {test_element2}%\n')

        plotting(plot1, plot2, plot3)
        
        user_save = input("\nSave this Prediction (Y/N): ").upper()
        if user_save == 'N':
            continue
        elif user_save == 'Y':
            save_to_csv(last_element1, test_element2, ticker)
        
        user_input = input("\nTry anoter Crypto? (Y/N): ").upper()
        if user_input == 'N':
            break
        elif user_input == 'Y':
            continue
        
    