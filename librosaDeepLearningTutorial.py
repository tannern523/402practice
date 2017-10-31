# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 21:34:00 2017

@author: Nick
"""

import IPython.display as ipd
# % pylab inline
from IPython import get_ipython
get_ipython().run_line_magic('pylab', 'inline')
import os
import pandas as pd
import librosa
import librosa.display
import glob

ipd.Audio('e:\\Nick\\Samples\\Yheti Clan Sample Pack\\Internet Kicks\\Internet Kick (8).wav')

y, sr = librosa.load('e:\\Nick\\Samples\\Yheti Clan Sample Pack\\Internet Kicks\\Internet Kick (8).wav')
print("Audio Buffer Size (y): ", len(y), " Sample Rate: ", sr)

plt.figure(figsize=(12,4))
librosa.display.waveplot(y, sr=sr)

