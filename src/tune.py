from src.model import Model, tune_hyperparameters
from src.simulation import simulate_patients
from src.utils import get_train_test_sequences

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# glucose threshold below which we detect the onset of hypoglycemia, in mg/dL
glucose_threshold = 54

# minimum length of a hypoglycemic event, in minutes
event_duration_threshold = 15

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=84,   # length of the time series, in days
    num=100,     # number of time series
)

# split the data into training and test sequences
training_sequences, test_sequences = get_train_test_sequences(
    data=data,
    time_worn_threshold=time_worn_threshold,
    glucose_threshold=glucose_threshold,
    event_duration_threshold=event_duration_threshold,
    test_size=0.3,
)

# find the best hyperparameters
parameters, score = tune_hyperparameters(
    sequences=training_sequences,
    n_splits=3,
    n_trials=5,
)

# fit the model to the training set
model = Model()

model.fit(
    sequences=training_sequences,
    l1_penalty=parameters['l1_penalty'],
    l2_penalty=parameters['l2_penalty'],
    learning_rate=parameters['learning_rate'],
    batch_size=parameters['batch_size'],
    epochs=parameters['epochs'],
    verbose=0
)

# evaluate the model on the test set
metrics = model.evaluate(sequences=test_sequences)
print(metrics)
