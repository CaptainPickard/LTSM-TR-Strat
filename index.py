from historic_data import get_user_crypto
from data_prep import format_data

if __name__ == '__main__':
    post_pross = get_user_crypto()
    
    pre_pross = format_data(post_pross)

    print('\n**Formatting Data Complete**\n')