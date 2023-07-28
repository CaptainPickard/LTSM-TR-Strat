from Historic_Crypto import Cryptocurrencies
import pandas as pd

crypto = Cryptocurrencies().find_crypto_pairs()

df = pd.DataFrame(crypto)
df.to_csv("crypto_pairs.csv", index="False")

print(crypto)