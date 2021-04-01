#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.multioutput import MultiOutputClassifier
from tsfresh import extract_features,select_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import xgboost as xgb



twelve_lead_model_filename = '12_lead_model.sav'
six_lead_model_filename = '6_lead_model.sav'
three_lead_model_filename = '3_lead_model.sav'
two_lead_model_filename = '2_lead_model.sav'
dxs = pd.read_csv('dx_mapping_scored.csv',usecols=['SNOMED CT Code'])['SNOMED CT Code'].astype(str).to_list()
dxs =sorted(dxs, key=lambda x: int(x))

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract labels from dataset.
    print('Extracting classes...')
    ylabels = np.zeros((num_recordings,27))
    for i,header_file in enumerate(header_files):
        header = load_header(header_file)
        labels = get_labels(header)
        for j,lab in enumerate(dxs):
            for r in labels:
                if str(r)==lab:ylabels[i,j]=1
    classes = set(dxs)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = [] 

    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        r_features = get_features(header, recording, twelve_leads)
        data.append(r_features)
    dftrain = pd.concat(data)
    dftrain.to_csv('extracted_features.csv',index=False)
    # Train models.

    # Define parameters for random forest classifier.
    param = {'n_estimators':1000,'max_depth': 6, 'eta': 0.05, 'objective': 'binary:logistic',
        'random_state':42,'eval_metric':'auc','tree_method':'hist','subsample':0.9
        }
    xgbmodel = xgb.XGBClassifier(**param)

    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    features = dftrain.to_numpy()
    classifier = MultiOutputClassifier(xgbmodel,n_jobs=-1).fit(features, ylabels)
    save_model(filename, classes, leads, classifier)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)
    names = [name for name in dftrain.columns if name.split('__',maxsplit=1)[0] in leads]
    names.extend(['sex','age'])
    features = dftrain[names].to_numpy()

    classifier = MultiOutputClassifier(xgbmodel,n_jobs=-1).fit(features, ylabels)
    save_model(filename, classes, leads, classifier)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)

    names = [name for name in dftrain.columns if name.split('__',maxsplit=1)[0] in leads]
    names.extend(['sex','age'])
    features = dftrain[names].to_numpy()

    classifier = MultiOutputClassifier(xgbmodel,n_jobs=-1).fit(features, ylabels)
    save_model(filename, classes, leads, classifier)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)

    names = [name for name in dftrain.columns if name.split('__',maxsplit=1)[0] in leads]
    names.extend(['sex','age'])
    features = dftrain[names].to_numpy()

    classifier = MultiOutputClassifier(xgbmodel,n_jobs=-1).fit(features, ylabels)
    save_model(filename, classes, leads, classifier)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads,  'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    r_features = get_features(header, recording, twelve_leads)
    names = [name for name in r_features.columns if name.split('__',maxsplit=1)[0] in leads]
    names.extend(['sex','age'])
    features = r_features[names].to_numpy().reshape(1,-1)


    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_recording_name(header):
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            recording_name = entries[0]
        else:
            break
    return recording_name

def to_align(freq, num_samples,recording,leads):
    recording = recording.T
    resampled=np.empty((500,len(leads))) #resmaple into 50 points for one second, for total 10 seconds
    start_index = np.argmax(recording[:freq,0])
    recording = recording[start_index:,:]
    resmaple_interval = 2*freq//50
    cycles = (num_samples-start_index)//freq
    print(cycles)
    if  cycles <10:
        num_cycle_tofill = (10 - cycles)//cycles
        if num_cycle_tofill > 0:
            remaning_tofill = 10*freq - (cycles + num_cycle_tofill)*freq
            temp = np.concatenate([recording[:cycles*freq,:] ]*(num_cycle_tofill+1))
            recording = np.concatenate([temp,recording[:(10*freq-temp.shape[0]),:]])
        
        else:
            recording = np.concatenate([recording[:cycles*freq,:],recording[:(10*freq-cycles*freq),:]])
       
    for i in range(1,251):
        for j in range(len(leads)):
            nmax = np.max(recording[(i-1)*resmaple_interval:i*resmaple_interval,j])
            nmax_index = np.argmax(recording[(i-1)*resmaple_interval:i*resmaple_interval,j])+(i-1)*resmaple_interval
            nmin = np.min(recording[(i-1)*resmaple_interval:i*resmaple_interval,j])
            nmin_index = np.argmin(recording[(i-1)*resmaple_interval:i*resmaple_interval,j])+(i-1)*resmaple_interval
            if nmax_index <= nmin_index:
                resampled[(i-1)*2,j]=nmax
                resampled[(i-1)*2+1,j]=nmin        
            else:
                resampled[(i-1)*2,j]=nmin
                resampled[(i-1)*2+1,j]=nmax  
    return resampled

def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    num_samples = int(get_num_samples(header))
    freq = int(get_frequency(header))
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]
    #align and resample the recording data
    recording = to_align(freq, num_samples,recording,available_leads)

    #Create dataframe
    to_extract={'friedrich_coefficients':[{'coeff': 1, 'm': 3, 'r': 30}],
            'number_crossing_m':[{'m': 0}],
            'change_quantiles':[{'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'}, 
                                
                                {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'}, 
                                
                                {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'}, 
                                
                                {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}, 
                                
                                {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'}, 
                                
                                {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, 
                                {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, 
                                
                                
                                
                                {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}, 
                                
                                {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, 
                                
                                {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, 
                                {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, 

                                {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, 
                                {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, 

                                {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, 
                                
                                {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}],
            'autocorrelation':[{'lag': 5}],
            'cwt_coefficients':[{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 1, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 2, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 3, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 4, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 2},
                                {'widths': (2, 5, 10, 20), 'coeff': 5, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 6, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 7, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 8, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 9, 'w': 20}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 10, 'w': 10}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 11, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 12, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 13, 'w': 10},
                                {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 2}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 5}, 
                                {'widths': (2, 5, 10, 20), 'coeff': 14, 'w': 10}],
              'agg_autocorrelation':[{'f_agg': 'mean', 'maxlag': 40}, {'f_agg': 'median', 'maxlag': 40}],
              'percentage_of_reoccurring_datapoints_to_all_datapoints':None,
              'percentage_of_reoccurring_values_to_all_values':None,
              'ratio_beyond_r_sigma':[ {'r': 2}],
              'quantile':[ {'q': 0.6}],
              'time_reversal_asymmetry_statistic':[{'lag': 1}, {'lag': 2}, {'lag': 3}],
              'fft_aggregated':[ {'aggtype': 'skew'}],
              'ratio_value_number_to_time_series_length':None,
              'permutation_entropy':[
                  {'tau': 1, 'dimension': 3}, 
                  {'tau': 1, 'dimension': 4}, 
                  {'tau': 1, 'dimension': 5}, 
                  {'tau': 1, 'dimension': 6}, 
                  {'tau': 1, 'dimension': 7}
              ],
              'number_peaks':[{'n': 1}],
              'agg_linear_trend':[ {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}, 
                                  {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'},],
              'minimum':None}
              
    df = pd.DataFrame(recording,columns=available_leads)
    df['recording_name']=get_recording_name(header)
    df_s = extract_features(df, column_id='recording_name',default_fc_parameters=to_extract)
    df_s['sex'] = sex
    df_s['age'] = age
    df_s=impute(df_s)
    return df_s