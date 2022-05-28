import os
import time
import numpy as np
import librosa
import librosa.display, librosa.decompose
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks

# process ok so milestones
# non-live version then live version
# load audio, split into 3 second chunks
# analyse those chunks one by one
# export scaled values (correlate to a random thing)

def split_audio(file_name, path):

    # separates audios into 3s files
    # saves as .wav files in a folder called 'chunked'

    path_create = os.path.join(path, 'chunked')
    os.makedirs(path_create) # creating a folder named chunked
    print('chunked')

    track = os.path.join(path, file_name)
    audio = AudioSegment.from_file(track, "mp3") 
    chunk_len = 3000 # pydub calculates in millisec 
    chunks = make_chunks(audio,chunk_len)
    for i, chunk in enumerate(chunks): 
        chunk_name = './chunked/' + file_name + "_{0}.wav".format(i) 
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav")

def dummy_analyse(file_name, path):

    # loads a chunk of audio for arbitry analysis
    # for now we're just going to use a value from the mel spectrum so that we're not going to have a massive discrepency between load times

    track = os.path.join(path, file_name)
    data, sr = librosa.load(track, res_type='kaiser_fast')

    spec = librosa.feature.melspectrogram(y=data, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    




