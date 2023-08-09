from historic_data import get_user_crypto
from data_prep import format_data
from PMP import mlm_model

if __name__ == '__main__':
    post_pross = get_user_crypto()
    x_pross = format_data(post_pross)
    mlm_model(x_pross)
    
    