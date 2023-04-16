from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_train_test_data

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# blood glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
blood_glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
episode_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=365,  # length of the time series, in days
    num=100,     # number of time series
)

# split the data into training and test sets
training_sequences, test_sequences = get_train_test_data(
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
    l1_penalty=0.005,
    l2_penalty=0.05,
    learning_rate=0.00001,
    batch_size=32,
    epochs=1000,
    verbose=0
)

# evaluate the model on the test set
metrics = model.evaluate(sequences=test_sequences)
