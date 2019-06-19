"""
=============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : generator_options.py
                    Contains all options for the generator class
    Author        : Bart van Erp
    Date          : 17/06/2019

==============================================================================
"""

## Set paths to data and list
path_data = "../Data/train/audio/"
path_data_test = "../Data/test/audio/"
path_list_test2 = "../Data/test/lists/test_set.txt"
path_list_train = "../Data/train/lists/train_set.txt"
path_list_test = "../Data/train/lists/test_set.txt"


## Specify classes to be distinguished between
labels=("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown")


# set options for training generator (speech)
train_speech_options = {
    "return_label": True,            # return label: True, False
    "data_type": "speech",           # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_train,    # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (16000,),         # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0.1                 # variance of additivie Gaussian noise
    }

# set options for training generator (spectrum)
train_spectrum_options = {
    "return_label": True,            # return label: True, False
    "data_type": "spectrum",         # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_train,    # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (99,161),         # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0.1                 # variance of additivie Gaussian noise
    }

# set options for training generator (mfcc)
train_mfcc_options = {
    "return_label": True,            # return label: True, False
    "data_type": "mfcc",             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_train,    # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (13,32),          # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0.1                 # variance of additivie Gaussian noise
    }

############TEST###############3

# set options for training generator (speech)
test_speech_options = {
    "return_label": True,            # return label: True, False
    "data_type": "speech",           # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_test,     # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (16000,),         # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                   # variance of additivie Gaussian noise
    }

# set options for test generator (spectrum)
test_spectrum_options = {
    "return_label": True,            # return label: True, False
    "data_type": "spectrum",         # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_test,     # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (99,161),         # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                   # variance of additivie Gaussian noise
    }

# set options for test generator (mfcc)
test_mfcc_options = {
    "return_label": True,            # return label: True, False
    "data_type": "mfcc",             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_test,     # path to list
    "batch_size": 64,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": (13,32),          # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                   # variance of additivie Gaussian noise
    }


################## TOTAL MODELS ####################3
# set options for test generator (mfcc)
train_all_options = {
    "return_label": True,            # return label: True, False
    "data_type": ["speech", "spectrum", "mfcc"],             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_train,     # path to list
    "batch_size": 32,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": [(16000,), (99,161), (13,32)],          # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "class",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "std",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                   # variance of additivie Gaussian noise
    }

val_all_options = {
    "return_label": True,            # return label: True, False
    "data_type": ["speech", "spectrum", "mfcc"],             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data,          # path to data folder
    "list_path": path_list_test,     # path to list
    "batch_size": 32,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": [(16000,), (99,161), (13,32)],          # output dimensions as tuple
    "shuffle_on_end": True,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "sample",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "95",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                 # variance of additivie Gaussian noise
    }

test_all_options = {
    "return_label": False,            # return label: True, False
    "data_type": ["speech", "spectrum", "mfcc"],             # type of data output: "speech", "spectrum", "mfcc"
    "data_path": path_data_test,          # path to data folder
    "list_path": path_list_test2,     # path to list
    "batch_size": 6,                # number of data segments per batch
    "labels": labels,                # list of labels
    "dims_data": (16000,),           # dimensions of original data (+zero append)
    "dims_output": [(16000,), (99,161), (13,32)],          # output dimensions as tuple
    "shuffle_on_end": False,          # shuffle data order per epoch: True, False
    "VAD": True,                     # apply voice activity detection 
    "downsample": 1,                 # downsample factor (disable: 1)
    "normalize_class": "sample",      # normalize sample with respect to "class", "sample"
    "subtract_mean": True,           # subtract the mean of the signal
    "normalize_type": "95",         # normalize the signal by dividing by None, "std" (class) or None, "std", "var", "95", "98" (sample)
    "normalize_eps": 1e-8,           # constant to prevent division by zero
    "noise_var": 0                 # variance of additivie Gaussian noise
    }