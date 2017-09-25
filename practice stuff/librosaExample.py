# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline

#
# librosa.core
#

y_orig, sr_orig = librosa.load(librosa.util.example_audio_file(), sr=None)
#y_orig, sr_orig = librosa.load('e:\\Nick\\Samples\\Yheti Clan Sample Pack\\Internet Kicks\\Internet Kick (8).wav', sr=None)
#y_orig, sr_orig = librosa.load('e:\\Nick\\Samples\\Yheti Clan Sample Pack\\91 Yheti Osc Samples\\Osc1.wav', sr=None)
print("Original Buffer Size (y): ", len(y_orig), "Original Sample Rate (sr): ", sr_orig)

sr = 22050
y = librosa.resample(y_orig, sr_orig, sr)
print("Resampled y and sr: ", len(y), sr)

IPython.display.Audio(data=y, rate=sr)

print(librosa.samples_to_time(len(y), sr), "sec")

# Spectral Representations
D = librosa.stft(y)
print(D.shape, D.dtype)

S, phase = librosa.magphase(D)
print(S.dtype, phase.dtype, np.allclose(D, S * phase))

# Constant-Q Transform
C = librosa.cqt(y, sr=sr)
print(C.shape, C.dtype)

#
# librosa.feature
#

melspec = librosa.feature.melspectrogram(y=y, sr=sr)
melspec_stft = librosa.feature.melspectrogram(S=S**2, sr=sr)
print(np.allclose(melspec, melspec_stft))

chroma = librosa.feature.chroma_stft(y=y, sr=sr)

#
# librosa.display
#

# Waveform Display
plt.figure()
librosa.display.waveplot(y=y, sr=sr)

# A Basic Spectogram Display
plt.figure()
librosa.display.specshow(melspec, y_axis='log', x_axis='time')
plt.colorbar()

