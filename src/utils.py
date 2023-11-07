import itertools
import pandas as pd
import numpy as np

def get_event_durations(glucose, glucose_threshold):
    '''
    Get the durations of the hypoglycemic events of a given patient, in minutes.
    '''
    
    # get the frequency of the data, in minutes
    freq = glucose.index.to_series().diff().mode()[0].total_seconds() // 60
    
    # get the event indicators
    x = np.logical_and(glucose < glucose_threshold, ~np.isnan(glucose)).astype(int)
    
    # get the event durations
    return np.array([len(x) * freq for x in [list(group) for key, group in itertools.groupby(x)] if 1 in x], dtype=np.int32)


def get_sequence_label(glucose, event_duration_threshold, glucose_threshold):
    '''
    Label a patient's one-week subsequence as 1 if the patient experienced
    a hypoglycemic event during the week and as 0 otherwise.
    '''
    
    # get the durations of the patient's hypoglycemic events over the given week
    d = get_event_durations(glucose, glucose_threshold)
    
    # check if the patient had any hypoglycemic events over the given week
    if len(d) > 0:
        
        # check if the length of any hypoglycemic event was above the threshold
        if np.max(d) >= event_duration_threshold:
            
            # if the length of any event was above the threshold, label the sequence as 1
            return 1
        
        # if the length of all events was below the threshold, label the sequence as 0
        else:
            return 0
    
    # if the patient didn't have any hypoglycemic event, label the sequence as 0
    else:
        return 0


def get_labelled_sequences(data, time_worn_threshold, glucose_threshold, event_duration_threshold):
    '''
    Get the labelled sequences as a list of dictionaries with the following items:

        patient: str.
            The patient id.

        start: pd.datetime.
            The first timestamp of the subsequent week.

        end: pd.datetime.
            The last timestamp of the subsequent week.

        X: np.ndarray.
            The patient's glucose measurements over the current week.

        L: np.ndarray.
            The patient's number of non-missing glucose measurements over the current week.

        Y: np.ndarray.
            A binary class label indicating whether the patient experienced a hypoglycemic event
            over the subsequent week (Y = 1) or not (Y = 0).
    '''
    
    # get the frequency of the data, in minutes
    minutes = int(data.index.to_series().diff().mode()[0].total_seconds() // 60)
    
    # calculate the number of timestamps in one week
    sequence_length = int(7 * 24 * 60 // minutes)
    
    # create a list for storing the data
    sequences = []
    
    # loop across the patients
    for patient in data.columns:
        
        # loop across the dates
        for t in range(sequence_length, (data.shape[0] // sequence_length) * sequence_length, sequence_length):
            
            # extract the patient's data over the current week
            X = data[patient].iloc[t - sequence_length: t]
            
            # extract the patient's data over the subsequent week
            Y = data[patient].iloc[t: t + sequence_length]
            
            # check if the patient has worn the device for a sufficient time over both weeks
            if pd.notna(X).mean() >= time_worn_threshold and pd.notna(Y).mean() >= time_worn_threshold:
                
                # get the input sequence
                X = X.dropna().to_list()
                
                # get the length of the input sequence
                L = len(X)
                
                # derive the class label
                Y = get_sequence_label(Y, event_duration_threshold, glucose_threshold)
                
                # save the patient's data
                sequences.append({
                    'patient': patient,
                    'start': str(data.index[t - 1] + pd.Timedelta(minutes=minutes)),
                    'end': str(data.index[t - 1] + pd.Timedelta(days=7)),
                    'X': X,
                    'L': L,
                    'Y': Y,
                })
    
    return sequences


def get_unlabelled_sequences(data, time_worn_threshold):
    '''
    Get the unlabelled sequences as a list of dictionaries with the following items:

        patient: str.
            The patient id.

        start: pd.datetime.
            The first timestamp of the subsequent week.

        end: pd.datetime.
            The last timestamp of the subsequent week.

        X: np.ndarray.
            The patient's glucose measurements over the current week.

        L: np.ndarray.
            The patient's number of non-missing glucose measurements over the current week.
    '''
    
    # get the frequency of the data, in minutes
    minutes = int(data.index.to_series().diff().mode()[0].total_seconds() // 60)
    
    # calculate the number of timestamps in one week
    sequence_length = int(7 * 24 * 60 // minutes)
    
    # create a list for storing the data
    sequences = []
    
    # loop across the patients
    for patient in data.columns:
            
            # extract the patient's data over the current week
            X = data[patient].iloc[- sequence_length:]
            
            # check if the patient has worn the device for a sufficient time over the current week
            if pd.notna(X).mean() >= time_worn_threshold:
                
                # get the input sequence
                X = X.dropna().to_list()
                
                # get the length of the input sequence
                L = len(X)
                
                # save the patient's data
                sequences.append({
                    'patient': patient,
                    'start': str(data.index[- 1] + pd.Timedelta(minutes=minutes)),
                    'end': str(data.index[- 1] + pd.Timedelta(days=7)),
                    'X': X,
                    'L': L,
                })
    
    return sequences
