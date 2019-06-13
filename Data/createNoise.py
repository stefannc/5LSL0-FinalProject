import numpy as np
import scipy.io.wavfile as wav
import os

#Global variables
TIME = 1
FS = 16000
N_SAMPLES = TIME * FS
MU_MU = -0.1644 #mean of mean
SIGMA_MU = 2022.657 #mean of std
MU_SIGMA = 2.951 #std of mean
SIGMA_SIGMA = 151.505 #std of std
n_output = 1000 #How many files there should be

def generate(mu, sigma):
    x = np.random.normal(mu, sigma, size = N_SAMPLES)
    return x

if 'noise' not in os.listdir('train/audio/'):
    print('No noise folder found in data, folder will be created')
    os.mkdir('Data/train/audio/noise') 
    print('Noise folder is created')
else:
    count = len(os.listdir('train/audio/noise/'))
    if count == n_output:
        print(n_output, 'noise files found. No new ones will be created')
    else:
        create = True
        n_needed = n_output - count
        print(count, 'noise files found.', n_needed, 'files will be created.')

means = np.random.normal(MU_MU, MU_SIGMA, size = n_needed)
stds = np.random.normal(SIGMA_MU, SIGMA_SIGMA, size = n_needed)

for i in range(count, n_output):
    noise = generate(means[i-count], stds[i-count])
    noise.astype(np.int16)
    name = 'noise_' + str(i+1) + '.wav'
    wav.write('train/audio/noise/' + name, FS, noise)

if create:
    print('Successfully created', i+1, 'noise .wav files')