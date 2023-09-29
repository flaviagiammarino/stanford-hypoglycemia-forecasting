from src.model import Model
from src.simulation import simulate_patients
from src.utils import get_inference_data

# minimum percentage of time that the patient must have worn the device over a given week
time_worn_threshold = 0.7

# generate some dummy data
data = simulate_patients(
    freq=5,      # sampling frequency of the time series, in minutes
    length=7,    # length of the time series, in days
    num=100,     # number of time series
)

# split the data into sequences
sequences = get_inference_data(
    data=data,
    time_worn_threshold=time_worn_threshold,
)

# load the model
model = Model()
model.load(directory='model')

# generate the model predictions
predictions = model.predict(sequences=sequences)

print(predictions.head(10))
#    patient                start                  end  predicted_label  predicted_probability  decision_threshold
# 0        0  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.094029                0.45
# 1        1  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.119137                0.45
# 2        2  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.046282                0.45
# 3        3  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.090396                0.45
# 4        4  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.126644                0.45
# 5        5  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.486400                0.45
# 6        6  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.077495                0.45
# 7        7  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.990677                0.45
# 8        8  2023-09-29 00:00:00  2023-10-05 23:55:00                0               0.083267                0.45
# 9        9  2023-09-29 00:00:00  2023-10-05 23:55:00                1               0.999524                0.45