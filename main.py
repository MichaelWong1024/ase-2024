import numpy as np
from scipy.io import wavfile

Fs, wavData = wavfile.read('sweep.wav')

if (wavData.dtype != np.float32):
	wavData = wavData / np.iinfo(wavData.dtype).max

np.savetxt('output2.txt', wavData, fmt='%.8f')
