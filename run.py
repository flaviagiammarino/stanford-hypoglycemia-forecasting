from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of clinically significant hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a clinically significant hypoglycemic episode, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in number of minutes
    length=365,  # length of the time series, in number of days
    num=100,     # number of time series
)

# split the data into training and test sets
training_sequences, test_sequences = get_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    blood_glucose_threshold=blood_glucose_threshold,
    episode_duration_threshold=episode_duration_threshold,
    test_size=0.2,
)

# fit the model to the training set
model = Model()

model.fit(
    sequences=training_sequences,
    num_features=10000,
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.00001,
    batch_size=32,
    epochs=500,
    verbose=0
)

# generate the training set predictions
training_results = model.predict(sequences=training_sequences)

# generate the test set predictions
test_results = model.predict(sequences=test_sequences)

