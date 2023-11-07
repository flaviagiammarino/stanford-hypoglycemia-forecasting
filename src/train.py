from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_labelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
event_duration_threshold = 15

# generate a dummy dataset
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# reshape the dataset from long to wide
data = data.pivot(index='ts', columns=['id'], values=['gl'])
data.columns = data.columns.get_level_values(level='id')

# split the dataset into sequences
sequences = get_labelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    glucose_threshold=glucose_threshold,
    event_duration_threshold=event_duration_threshold,
)

# train the model
model = Model()

model.fit(
    sequences=sequences,
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.00001,
    batch_size=32,
    epochs=1000,
    verbose=1
)

# save the model
model.save(directory='model')
