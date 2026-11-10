import pandas as pd
import numpy as np

df = pd.read_csv("data/Stage1_Train.csv", dtype=str, keep_default_na=False)
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Target counts:")
print(df['y'].value_counts())
