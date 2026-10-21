import os
import pandas as pd

def ensure_data_path():
    if not os.path.exists('data'):
        os.makedirs('data')

def load_raw(path='data/fake_and_real_news.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)
