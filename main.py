import pandas as pd

raw_csv_data = pd.read_csv('Absenteeism_data.csv')
# print(raw_csv_data)

df = raw_csv_data.copy()
df.drop(['ID'], 1, inplace=True)
df_no_age = df.drop(['Age'], 1)

reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
reason_columns['check'] = reason_columns.sum(1)
reason_columns = reason_columns.drop(['check'], 1)
age_dummies = pd.get_dummies(df['Age'])
df.drop(['Reason for Absence'], 1, inplace=True)

reason_type_1 = reason_columns.loc[:, 1:14].max(1)
reason_type_2 = reason_columns.loc[:, 15:17].max(1)
reason_type_3 = reason_columns.loc[:, 18:21].max(1)
reason_type_4 = reason_columns.loc[:, 22:28].max(1)

df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], 1)

column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                'Daily Work Load Average', 'Body Mass Index', 'Education',
                'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df.columns = column_names
# print(df)

df_concatenated = pd.concat([df_no_age, age_dummies], 1)
# print(df_concatenated.head())


column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
                          'Distance to Work', 'Age',
                          'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']

df = df[column_names_reordered]

df_checkpoint = df_concatenated.copy()

df_checkpoint['Date'] = pd.to_datetime(df_checkpoint['Date'], format='%d/%m/%Y')

list_months = []

for i in range(df_checkpoint.shape[0]):
    list_months.append(df_checkpoint['Date'][i].month)

# print(list_months)

df_checkpoint['Month Value'] = list_months


def date_to_weekday(date_value):
    return date_value.weekday()


df_checkpoint['Day of the week'] = df_checkpoint['Date'].apply(date_to_weekday)

df_checkpoint = df_checkpoint.drop(['Date'], 1)

column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the week',
                    'Transportation Expense', 'Distance to Work', 'Age',
                    'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                    'Pets', 'Absenteeism Time in Hours']

df_checkpoint = df_checkpoint[column_names_upd]

df_checkpoint_v2 = df_checkpoint.copy()
df_checkpoint_v2['Education'] = df_checkpoint_v2['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

df_preprocessed = df_checkpoint_v2.copy()
print(df_preprocessed.head(20))
