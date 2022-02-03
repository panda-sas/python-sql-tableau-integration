from absenteeism_module import *
import pymysql

conn = pymysql.connect(database='predicted_outputs', user='root', password='Delta@Force9')
cursor = conn.cursor()
cursor.execute('SELECT * FROM predicted_outputs;')

insert_query = 'INSERT INTO predicted_outputs VALUES '


model = absenteeism_model('model', 'scaler')
model.load_and_clean_data('Absenteeism_new_data.csv')

# Feed the cleaned data into the model and deliver the output #

df_new_obs = model.predicted_outputs()
print(df_new_obs)

for i in range(df_new_obs.shape[0]):
    insert_query += '('

    for j in range(df_new_obs.shape[1]):
        insert_query += str(df_new_obs[df_new_obs.columns.values[j]][i]) + ', '

    insert_query = insert_query[:-2] + '), '

insert_query = insert_query[:-2] + ';'

cursor.execute(insert_query)
conn.commit()

conn.close()
