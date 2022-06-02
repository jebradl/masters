# aim is to have everything enough in one place so that i can just dump a file and run it

import os
import PIL
import pathlib
from matplotlib import image
import numpy as np
import pandas as pd
import librosa
import librosa.display
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

        self.image_height = 235
        self.image_width = 352

        track = file.replace('.mp3', '')
        self.track = track
        self.file = file

        self.parent = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/for_use/'
        self.folder_path = os.path.join(self.parent, self.track)
        self.folder_dir = os.path.join(self.folder_path) # is this what it should be?
        # does the below work when just using folder_path?

        try:
            os.makedirs(self.folder_dir)
        except:
            pass

        self.chunk_track()
        self.create_images_folder()
        
        for chunk in self.chunk_dir:
            self.create_image()
            self.get_scores()


    def chunk_track(self):
        folder_name = self.track + ' chunked'
        chunk_path = os.path.join(self.folder_path, folder_name)

        try:
            os.makedirs(chunk_path)
        except:
            pass

        self.chunk_dir = os.listdir(chunk_path)

        track_location = os.path.join(self.parent, self.file)
        audio = AudioSegment.from_file(track_location, 'mp3')
        chunk_len = 3000
        chunks = make_chunks(audio, chunk_len)

        for i, chunk in enumerate(chunks):
            chunk_name = self.track + "_{:02d}.wav".format(i) 
            chunk.export(chunk_path + '/' + chunk_name, format="wav")
    

    def create_images_folder(self):

        images_folder = self.track + ' images'
        self.images_folder = os.path.join(self.folder_path, images_folder)

        try:
            os.makedirs(self.images_folder)
        except:
            pass

    
    def create_image(self, chunk):

        # establish path? when you put it all together when does is make the joined path?
        data, sr = librosa.load(chunk, res_type='kaiser_fast')
        spec = librosa.feature.melspectrogram(y=data, sr=sr)
        spec_big = librosa.power_to_db(spec)

        image_name = chunk.split('.')
        image_name = image_name[0]

        img = librosa.display.specshow(spec_big)
        img_path = self.images_folder+"{}.png".format(image_name)
        plt.savefig(img_path, bbox_inches='tight')
        plt.clf()

        return img_path

    
    def get_scores(self, img_path):
        # change img size and not having to save then load img?
        # also scores=None type beat
        
        img = tf.keras.utils.load_img(
            img_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        scores = []
        for i in score:
            scores.append(float(i))
        
        return scores

    





        



track = 'the_heart.mp3'

analyse = SongAnalysis(track)