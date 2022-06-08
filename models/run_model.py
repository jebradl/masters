# aim is to have everything enough in one place so that i can just dump a file and run it

import os
import time
import pathlib
from matplotlib import image
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2 import adam

class SongAnalysis():

    def __init__(self,file,model_path=None):
        super().__init__()

        self.img_height = 235
        self.img_width = 352
        self.batch_size = 32

        self.genre_list = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

        track = file.replace('.mp3', '')
        self.track = track
        self.file = file

        if model_path != None:
            self.model_path = model_path

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
        self.load_model()

        # print(self.chunk_path)
        chunk_dir = os.listdir(self.chunk_path)
        # print("new:", chunk_dir)

        print("processing: start!")
        time.sleep(1)

        self.song_scores = []

        for chunk in chunk_dir:
            start = time.perf_counter()
            # print(chunk)
            # print(self.chunk_path)
            img = self.create_image(chunk)
            # self.generate_mean(chunk)
            score = self.get_scores(img)
            self.song_scores.append(score)
            remaining = 3-(time.perf_counter()-start)
            # time.sleep(remaining)

        self.process_results()


    def chunk_track(self):
        folder_name = self.track + ' chunked'
        chunk_path = self.folder_path + '/' + folder_name

        print(chunk_path)

        try:
            os.makedirs(chunk_path)
        except:
            pass

        # print(chunk_dir) 
        # self.chunk_dir = chunk_dir
        self.chunk_path = chunk_path

        # print(self.chunk_dir)

        track_location = os.path.join(self.parent, self.file)
        audio = AudioSegment.from_file(track_location, 'mp3')
        chunk_len = 3000
        chunks = make_chunks(audio, chunk_len)

        for i, chunk in enumerate(chunks):
            chunk_name = self.track + "_{:03d}.wav".format(i) 
            chunk.export(self.chunk_path + '/' + chunk_name, format="wav")
    

    def create_images_folder(self):

        images_folder = self.track + ' images'
        self.images_folder = self.folder_path + '/' + images_folder

        try:
            os.makedirs(self.images_folder)
        except:
            pass

    
    def create_image(self, chunk):

        # establish path? when you put it all together when does is make the joined path?
        chunk_path = os.path.join(self.chunk_path, chunk)
        data, sr = librosa.load(chunk_path, res_type='kaiser_fast')
        spec = librosa.feature.melspectrogram(y=data, sr=sr)
        spec_big = librosa.power_to_db(spec)

        image_name = chunk.split('.')
        image_name = image_name[0]

        img = librosa.display.specshow(spec_big)
        img_path = self.images_folder+"/{}.png".format(image_name)
        plt.savefig(img_path, bbox_inches='tight')
        plt.clf()

        return img_path

    def load_model(self):

        print("loading model...")

        self.model = keras.models.load_model(self.model_path)

        print("model loaded!")

    
    def get_scores(self, img_path):
        # change img size and not having to save then load img?
        # also scores=None type beat
        
        img = tf.keras.utils.load_img(
            img_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        self.scores = []
        for i in score:
            self.scores.append(float(i))

        # print(self.scores)
        return self.scores

    def generate_mean(self, file):
        
        track = os.path.join(self.chunk_path, file)
        x, sr = librosa.load(track, sr=None, mono=True)  # kaiser_fast

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

        feature = np.mean(f)
        print('song: {} mean: {}'.format(file, feature))

    def process_results(self):

        df = pd.DataFrame(self.song_scores, columns=self.genre_list)
        df.to_csv('data.csv')
        data = pd.read_csv('data.csv', index_col=False)

        data[self.genre_list].plot()
        # plt.show()

        plt.savefig(self.parent+'results/'+self.track, bbox_inches='tight')
        

    





track = 'this_side.mp3'
model = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/saved_models/model_v4'
analyse = SongAnalysis(track, model)