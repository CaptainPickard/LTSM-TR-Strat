from historic_data import get_user_crypto
from data_prep import format_data
from PMP import mlm_model
from plotting import plotting

if __name__ == '__main__':

    while True:
        post_pross, ticker = get_user_crypto()
        x_pross = format_data(post_pross)
        plot1, plot2, plot3 = mlm_model(x_pross)
        
        print(f"\n Asset: {ticker}\n")

        last_element1 = (plot3[-1])
        print(f'\n Tomorrows {ticker} Predicted Gain/loss: {last_element1}%\n')

        plotting(plot1, plot2, plot3)

        user_input = input("\nWould you like to try anoter Crypto? (Y/N): \n").upper()
        if user_input == 'N':
            break
        elif user_input == 'Y':
            continue
        
    