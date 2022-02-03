import pandas as pd
import numpy as np

# Import the Standard scalar module #
from sklearn.preprocessing import StandardScaler

# Import the train & test module #

# Splits arrays or matrices into random train and test subsets #
from sklearn.model_selection import train_test_split

# Logistic regression with sklearn #
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load the data #
data_preprocessed = pd.read_csv('df_preprocessed.csv')
# print(data_preprocessed.head())

# Create the targets #
# Two classes -> 'Moderately absent' and 'Excessively Absent'
# Taking the median value of the 'Absenteeism Time in Hours' and using it as a cut off line between two classes

targets = np.where(
    data_preprocessed['Absenteeism Time in Hours'] > data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

data_preprocessed['Excessive Absenteeism'] = targets
# print(data_preprocessed.head())

# Balancing the dataset -> What percentage of the targets are 1s and 0s? #
# print(targets.sum() / targets.shape[0])

# Checkpoint #
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'], 1)

# Selecting the inputs for the regression #
unscaled_inputs = data_with_targets.iloc[:, :-1]

# Standardize the data #
absenteeism_scaler = StandardScaler()
absenteeism_scaler.fit(unscaled_inputs)

scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
# print(scaled_inputs)


# Split the data into train & test and shuffle #

# Returns arrays of training dataset with inputs, training dataset with targets, test dataset with inputs and
# test dataset with targets -> So four variables to contain these arrays #
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# Training the model #
reg = LogisticRegression()
reg.fit(x_train, y_train)

# This method returns the mean accuracy on the given test data and labels #
reg.score(x_train, y_train)

# Manually check the accuracy #

# This method predicts class labels (logistics regression outputs) for given input samples #
model_outputs = reg.predict(x_train)

# Accuracy  = Correct predictions / Total Observations #
np.sum((model_outputs == y_train)) / model_outputs.shape[0]

# Finding the intercept and coefficients #
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)

summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)