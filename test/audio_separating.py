import librosa
import librosa.display
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
import numpy as np
import os

 

def process_audio(file_name, path):
    track = os.path.join(path, file_name)
    print(track)
    track_no = file_name.split('.')
    chunk_folder = track_no[0] + '-chunked'
    chunk_path = os.path.join(path, chunk_folder)
    print(chunk_path)
    audio = AudioSegment.from_file(track, "mp3") 
    chunk_len = 3000 # pydub calculates in millisec 
    # what about to the beat
    chunks = make_chunks(audio, chunk_len)
    for i, chunk in enumerate(chunks): 
        chunk_name = track_no[0] + "_{:02d}.wav".format(i) 
        print ("exporting", chunk_name) 
        chunk.export(chunk_path + '/' + chunk_name, format="wav")

path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds'
all_files = os.listdir(path)


# get a file from the samples
# create a chunked folder for that file
# put chunks in that folder
# analyse each of the chunks


for file in all_files:
    if ('.mp3' in file):
        try:
            # print('hello')
            track_no = file.split('.')
            folder_name = track_no[0] + '-chunked'
            print(folder_name)
            path_create = os.path.join(path, folder_name)
            os.makedirs(path_create) # creating a folder named chunked
            # print('chunked')
        except:
            pass

for file in all_files:
    if ('.mp3' in file):
        # file is just file
        # path is path to original file
        # destination is path + 'song-chunked'
        process_audio(file, path)
