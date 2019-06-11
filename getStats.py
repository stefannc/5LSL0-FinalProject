import numpy as np
import os
from scipy import signal
from scipy.io import wavfile

PATH = os.getcwd()
data_path = PATH + '/Data/train/audio/'

#Iterating over standard classes
classes = ['yes', 'no', 'up', 'down', 'left', 'right',
           'on', 'off', 'stop', 'go']

mean_per_class = []
std_per_class = []
for c in classes:
    print(c)
    class_path = data_path + '/' + c + '/'
    class_mean = []
    class_std = []
    for filename in os.listdir(class_path):
        _, x = wavfile.read(str(class_path) + str(filename))
        class_mean.append(np.mean(x))
        class_std.append(np.std(x))
    mean_per_class.append(np.mean(class_mean))
    std_per_class.append(np.mean(class_std))

#Iterating over unknown classes
class_mean = []
class_std = []
for c in os.listdir(data_path):
    if ((c in classes) or (c == '_background_noise_')):
        pass
    else:
        class_path = data_path + '/' + c + '/'
        for filename in os.listdir(class_path):
            _, x = wavfile.read(str(class_path) + str(filename))
            class_mean.append(np.mean(x))
            class_std.append(np.std(x))
mean_unknown = np.mean(class_mean)
std_unknown = np.mean(class_std)