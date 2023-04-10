import itertools
import pandas as pd
import numpy as np

def get_episodes_duration(blood_glucose,
                          blood_glucose_threshold):
    '''
    Get the durations of the hypoglycemic episodes of a given patient, in minutes.
    '''
    
    # get the frequency of the data, in minutes
    freq = blood_glucose.index.to_series().diff().mode()[0].total_seconds() // 60
    
    # get the episodes indicators
    x = np.logical_and(blood_glucose < blood_glucose_threshold, ~np.isnan(blood_glucose)).astype(int)
    
    # get the episodes durations
    return np.array([len(x) * freq for x in [list(group) for key, group in itertools.groupby(x)] if 1 in x], dtype=np.int32)


def get_sequence_label(blood_glucose,
                       episode_duration_threshold,
                       blood_glucose_threshold):
    '''
    Label a patient's one-week subsequence as 1 if the patient had a severe hypoglycemic
    episode during the week and as 0 otherwise.
    '''
    
    # get the durations of the patient's hypoglycemic episodes over the given week
    d = get_episodes_duration(blood_glucose, blood_glucose_threshold)
    
    # check if the patient had any hypoglycemic episodes over the given week
    if len(d) > 0:
        
        # check if the length of any hypoglycemic episodes was above the threshold
        if np.max(d) >= episode_duration_threshold:
            
            # if the length of any episode was above the threshold, label the sequence as 1
            return 1
        
        # if the length of all episodes was below the threshold, label the sequence as 0
        else:
            return 0
    
    # if the patient didn't have any hypoglycemic episodes, label the sequence as 0
    else:
        return 0


def get_sequences(data,
                  time_worn_threshold,
                  blood_glucose_threshold,
                  episode_duration_threshold,
                  test_size):
    '''
    Get the training and test data as lists of dictionaries with the following items:

        patient: str.
            The patient id.

        start: pd.datetime.
            The first timestamp of the subsequent week.

        end: pd.datetime.
            The last timestamp of the subsequent week.

        X: np.ndarray.
            The patient's blood glucose measurements over the current week.

        L: np.ndarray.
            The patient's number of non-missing blood glucose measurements over the current week.

        Y: np.ndarray.
            A binary class label indicating whether the patient had a severe hypoglycemic episode
            over the subsequent week (Y = 1) or not (Y = 0).
    '''
    
    # get the frequency of the data, in number of minutes
    minutes = int(data.index.to_series().diff().mode()[0].total_seconds() // 60)
    
    # calculate the number of timestamps in one week
    sequence_length = int(7 * 24 * 60 // minutes)
    
    # create a list for storing the training data
    training_sequences = []
    
    # create a list for storing the test data
    test_sequences = []
    
    # loop across the patients
    for patient in data.columns:
        
        # create a list for storing the patient's data
        sequences = []
        
        # loop across the dates
        for t in range(sequence_length, (data.shape[0] // sequence_length) * sequence_length, sequence_length):
            
            # extract the patient's data over the previous week
            X = data[patient].iloc[t - sequence_length: t]
            
            # extract the patient's data over the subsequent week
            Y = data[patient].iloc[t: t + sequence_length]
            
            # check if the patient has worn the device for a sufficient time over both weeks
            if pd.notna(X).mean() >= time_worn_threshold and pd.notna(Y).mean() >= time_worn_threshold:
                
                # get the input sequence
                X = X.dropna().to_list()
                
                # calculate the lengths of the input sequence
                L = len(X)
                
                # derive the class label
                Y = get_sequence_label(Y, episode_duration_threshold, blood_glucose_threshold)
                
                # save the patient's data
                sequences.append({
                    'patient': patient,
                    'start': str(data.index[t - 1] + pd.Timedelta(minutes=minutes)),
                    'end': str(data.index[t - 1] + pd.Timedelta(days=7)),
                    'X': X,
                    'L': L,
                    'Y': Y,
                })
        
        # split the patient's data
        training_sequences.extend(sequences[:int(round((1 - test_size) * len(sequences)))])
        test_sequences.extend(sequences[int(round((1 - test_size) * len(sequences))):])
    
    return training_sequences, test_sequences
