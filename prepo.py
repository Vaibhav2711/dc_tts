# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm

# Load data
fpaths, _, _ = load_data() # list
os.mkdir("/content/drive/My Drive/LJnaval/mels")
os.mkdir("/content/drive/My Drive/LJnaval/mags")
for fpath in tqdm.tqdm(fpaths):
    fname, mel, mag = load_spectrograms(fpath)
    #if not os.path.exists("mels"): os.mkdir("mels")
    #if not os.path.exists("mags"): os.mkdir("mags")

    np.save("/content/drive/My Drive/LJnaval/mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("/content/drive/My Drive/LJnaval/mags/{}".format(fname.replace("wav", "npy")), mag)
