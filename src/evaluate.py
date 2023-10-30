from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_labelled_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_labelled_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# evaluate the model
metrics = model.evaluate(sequences=sequences)
print(metrics)
