# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os
import shutil

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
from pydub import AudioSegment
#from pydub.playback import play

def synthesize():
    # Load data
    print("Input a sentence")
    sentence = input()
    sentence = " "+sentence
    L = load_data(sentence,"synthesize")

    # Load graph
    g = Graph(mode="synthesize"); print("Graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in tqdm(range(hp.max_T)):
            _gs, _Y, _max_attentions, _alignments = \
                sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)
        one_sec_segment = AudioSegment.silent(duration=1500)
        wavs = list(os.listdir(hp.sampledir))
        print(wavs)
        wavs.sort()
        print(wavs)
        audio_seg = [AudioSegment.from_file(file= hp.sampledir+"/"+wav, format="wav") for wav in wavs ]
        final_audio = AudioSegment.silent(duration=2000)
        #audio_seg.pop(0)
        for audio in audio_seg:
          final_audio = final_audio+audio
        final_audio = final_audio+AudioSegment.silent(duration=2000)
        final_audio = final_audio + 5
        final_audio.export(out_f = "Output.wav", 
                       format = "wav")
        shutil.rmtree(hp.sampledir)
         
if __name__ == '__main__':
    synthesize()
    print("Done")


