import numpy as np
import scipy.io.wavfile as wav
import os

def generate(mu, sigma):
    TIME = 1 #sec
    FS = 16000 #Hz
    N_SAMPLES = TIME * FS

    x = np.random.normal(mu, sigma, size = N_SAMPLES)
    return x

def create(n):
    MU_MU = -0.1644 #mean of mean
    SIGMA_MU = 2022.657 #mean of std
    MU_SIGMA = 2.951 #std of mean
    SIGMA_SIGMA = 151.505 #std of std
    FS = 16000 #Hz
    
    if 'silence' not in os.listdir('../Data/train/audio/'):
        print('No silence folder found in data, folder will be created')
        os.mkdir('../Data/train/audio/silence') 
        print('Silence folder is created')
        n_needed = n
        count = 0
    else:
        count = len(os.listdir('../Data/train/audio/silence/'))
        if count == n:
            print(n, 'silence files found. No new ones will be created')
            return
        else:
            create = True
            n_needed = n - count
            print(count, 'silence files found.', n_needed, 'files will be created.')
    
    means = np.random.normal(MU_MU, MU_SIGMA, size = n_needed)
    stds = np.random.normal(SIGMA_MU, SIGMA_SIGMA, size = n_needed)
    
    for i in range(count, n):
        noise = generate(means[i-count], stds[i-count])
        noise.astype(np.int16)
        name = 'silence_' + str(i+1) + '.wav'
        wav.write('../Data/train/audio/silence/' + name, FS, noise)
    
    if create:
        print('Successfully created', i+1, 'silence .wav files')
    return