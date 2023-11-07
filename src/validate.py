import pandas as pd
from sklearn.model_selection import StratifiedKFold

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

# split the sequences into folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# create a list for storing the results for each fold
results = []

# loop across the folds
for i, (train_index, test_index) in enumerate(skf.split(X=[s['X'] for s in sequences], y=[s['Y'] for s in sequences])):
    
    # fit the model to the training set
    model = Model()

    model.fit(
        sequences=[sequences[i] for i in train_index],
        l1_penalty=0.005,
        l2_penalty=0.05,
        learning_rate=0.00001,
        batch_size=32,
        epochs=1000,
        verbose=0
    )

    # evaluate the model on the test set
    metrics = model.evaluate(sequences=[sequences[i] for i in test_index])

    # save the results
    results.append(metrics)
    
    # display the results
    print('------------------------------------')
    print(f'fold {i + 1} of {skf.n_splits}:')
    print('------------------------------------')
    for k, v in metrics.items():
        print(f'{k}: {format(v, ".4f")}')

# organize the results in a data frame
results = pd.DataFrame(results)
print(results)

# average the results
print(results.mean())
