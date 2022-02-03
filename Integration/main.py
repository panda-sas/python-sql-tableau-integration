
from absenteeism_module import *

model = absenteeism_model('model', 'scaler')
model.load_and_clean_data('Absenteeism_new_data.csv')

# Feed the cleaned data into the model and deliver the output #
model.predicted_outputs()