# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset = pd.read_csv('breastCancer.csv')
print(dataset.head())
print(dataset.dtypes)
# Data preprocessing phase

data_bare_nucleoli = pd.DataFrame(dataset.bare_nucleoli)
#Replacing all Non integer value with NAN value
for i in dataset.bare_nucleoli.values:
    if str(i).isdigit():
        continue
    else:
        dataset.bare_nucleoli = dataset.bare_nucleoli.replace(i, np.nan)

#Filling all NAN values with the median value
dataset.bare_nucleoli = dataset.fillna(int(dataset.bare_nucleoli.median()))
#Changing the data type to int64 for bare_nucleoli field
dataset.bare_nucleoli = dataset.bare_nucleoli.astype('int64')
print(dataset.dtypes)
