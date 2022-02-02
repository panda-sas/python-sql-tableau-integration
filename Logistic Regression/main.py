import pandas as pd
import numpy as np

# Load the data #
data_preprocessed = pd.read_csv('df_preprocessed.csv')
# print(data_preprocessed.head())

# Create the targets #
# Two classes -> 'Moderately absent' and 'Excessively Absent'
# Taking the median value of the 'Absenteeism Time in Hours' and using it as a cut off line between two classes

targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

data_preprocessed['Excessive Absenteeism'] = targets
# print(data_preprocessed.head())

# Balancing the dataset -> What percentage of the targets are 1s and 0s? #
print(targets.sum() / targets.shape[0])

# Checkpoint #
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'], 1)

