from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_unlabelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# generate a dummy dataset
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=7,    # length of the time series, in days
    num=100,     # number of time series
)

# reshape the dataset from long to wide
data = data.pivot(index='ts', columns=['id'], values=['gl'])
data.columns = data.columns.get_level_values(level='id')

# split the dataset into sequences
sequences = get_unlabelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# generate the model predictions
predictions = model.predict(sequences=sequences)
print(predictions.head())
