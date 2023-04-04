import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_sequences, split_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.5

# blood glucose threshold below which we detect the onset of severe hypoglycemia
blood_glucose_threshold = 54

# minimum length of a severe hypoglycemic episode, in number of minutes
episode_duration_threshold = 20

# frequency of the time series, in number of minutes
sampling_frequency = 5

# generate some dummy data
data = simulate_patients(
    freq=sampling_frequency,   # sampling frequency of the time series, in minutes
    length=360,                # length of the time series, in days
    num=100,                   # number of time series
    distributed=True
)

# cast the columns to the respective data types
data['id'] = data['id'].astype(str)
data['datetime'] = pd.to_datetime(data['datetime'], infer_datetime_format=True, errors='coerce').dt.tz_localize(None)
data['glucose'] = data['glucose'].astype(float)

# reshape the data frame from long to wide
data = data.set_index('datetime').groupby(by='id')['glucose'].resample(f'{sampling_frequency}T').last().reset_index()
data = data.pivot(index='datetime', columns=['id'], values=['glucose'])
data.columns = data.columns.get_level_values(level='id')

# split the data into sequences
sequences = get_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
)

# split the sequences into training and test
training_sequences, test_sequences = split_sequences(
    sequences=sequences,
    test_size=0.3,
)

# fit the model to the training set
model = Model()

model.fit(
    sequences=training_sequences,
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.0001,
    batch_size=64,
    epochs=500,
    verbose=0
)

training_results = model.predict(sequences=training_sequences)

# evaluate the model on the test set
test_results = model.predict(sequences=test_sequences)
