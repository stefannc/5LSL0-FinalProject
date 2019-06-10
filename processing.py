import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd

# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

# Downsampling function
# Function that samples the input samples with sample_rate to a new sample rate, and returns the new samples
def downsampling(samples, sample_rate, new_sample_rate):
    return signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))

# VAD function
# Shortens the samples to only contain the part that has a value above 0.3 times the mean of the file
def VAD(samples, sample_rate):
    plt.figure(figsize=(14, 8))
    x_axis = np.linspace(0, sample_rate, len(samples))
    plt.plot(x_axis, samples)
    mean = np.mean(abs(samples))
    first = np.argmax(abs(samples)>0.3*mean)
    last = int(np.argmax(abs(samples[::-1])>0.3*mean))
    last = len(samples) - last
    plt.plot(x_axis[first:last], samples[first:last])
    plt.savefig('figures/VAD.pdf')
    return samples[first:last]

# Padding function
# Adds zero's to the samples in order to make sure all blocks have the same length
def padder(samples, req_len):
    if(len(samples) < req_len):
        samples = np.pad(samples, (0, req_len-len(samples)), 'constant')
    return samples

# Spectrogram function
# Function that computes the spectrogram of samples at a certain sample rate
# Returns frequencies, times and the logaritmic magnitude of the samples
def log_specgram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

# MFCC function
def MFCC(samples, sample_rate):
    Mel = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(Mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc

# INIT
time_duration = 0.8 # In seconds

# Read files
train_audio_path = 'train/audio'
filename = '/zero/0b56bcfe_nohash_1.wav'

sample_rate, samples = wavfile.read(str(train_audio_path) + filename)

# Downsample
new_sample_rate = int(8e3)
samples = downsampling(samples, sample_rate, new_sample_rate)

# VAD
samples = VAD(samples, new_sample_rate)

# Zero padding
req_len = new_sample_rate
samples = padder(samples, req_len)

# Plot timedomain
x_axis = np.linspace(0, new_sample_rate, len(samples))
plt.figure(figsize=(14, 8))
plt.plot(x_axis, samples)
plt.savefig('figures/time_domain.pdf')

# Spectrogram
freqs, times, spectrogram = log_specgram(samples, new_sample_rate)
x_axis = np.linspace(0, new_sample_rate, len(samples))

# Plot spectrogram
plt.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[times.min(), times.max(), freqs.min(), freqs.max()])
plt.savefig('figures/spectral_domain.pdf')

# MFCC
mfcc = MFCC(samples, sample_rate)
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()
plt.savefig('figures/mfcc_domain.pdf')