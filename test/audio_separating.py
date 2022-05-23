import librosa
import librosa.display
from pydub import AudioSegment
from pydub.utils import make_chunks
import matplotlib.pyplot as plt
import numpy as np
import os

# data, sr = librosa.load('c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds/next_level.mp3', res_type='kaiser_fast')

# plt.figure(figsize=(12,4))
# librosa.display.waveshow(data, sr=sr)
# plt.show()

# file_path 


#### this is doing the thing but creating them in the wrong place 

def process_audio(file_name, path):
    track = os.path.join(path, file_name)
    audio = AudioSegment.from_file(track, "mp3") 
    chunk_len = 3000 # pydub calculates in millisec 
    chunks = make_chunks(audio,chunk_len)
    for i, chunk in enumerate(chunks): 
        chunk_name = './chunked/' + file_name + "_{0}.wav".format(i) 
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav") 

path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds'
all_files = os.listdir(path)

try:
    print('hello')
    path_create = os.path.join(path, 'chunked')
    os.makedirs(path_create) # creating a folder named chunked
    print('chunked')
except:
    print('no')
    pass
print('not again')
for file in all_files:
    print(file)
    if ('.mp3' in file):
        process_audio(file, path)
