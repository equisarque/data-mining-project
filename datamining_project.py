# Datamining Project Joris FELZINES - Guillaume MANTZARIDES


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_table("../../flickr_data2.csv", sep=",", low_memory=False)

print(data.head())
print(data.info())
print(data.describe())
