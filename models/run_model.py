# aim is to have everything enough in one place so that i can just dump a file and run it

import os
import PIL
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential



class SongAnalysis():

    def __init__(self,file):
        super().__init__()

        track = file.replace('.mp3', '')
        self.track = track
        self.file = file

        self.parent = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/for_use/'
        self.folder_path = os.path.join(self.parent, self.track)
        self.folder_dir = os.path.join(self.folder_path)

        try:
            os.makedirs(self.folder_dir)
        except:
            pass

        self.chunk_track()

        # print(self.track, self.folder_path)

    def chunk_track(self):
        folder_name = self.track + '-chunked'
        chunk_path = os.path.join(self.folder_path, folder_name)

        try:
            os.makedirs(chunk_path)
        except:
            pass

        self.chunk_dir = os.path.join(self.folder_path, folder_name)

        track_location = os.path.join(self.parent, self.file)
        audio = AudioSegment.from_file(track_location, 'mp3')
        chunk_len = 3000
        chunks = make_chunks(audio, chunk_len)

        for i, chunk in enumerate(chunks):
            chunk_name = self.track + "_{:02d}.wav".format(i) 
            chunk.export(chunk_path + '/' + chunk_name, format="wav")
    
    






        



track = 'the_heart.mp3'

analyse = SongAnalysis(track)