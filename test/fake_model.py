import os
import time
import librosa
import numpy as np

# editing the features.py file so that it generates a single value for mean of mel spec

def generate_mean(file, path):
    track = os.path.join(path, file)
    x, sr = librosa.load(track, sr=None, mono=True)  # kaiser_fast

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    feature = np.mean(f)
    print('song: {} mean: {}'.format(file, feature))


path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds'
all_files = os.listdir(path)


for file in all_files:
    if ('.mp3' in file):
        track_no = file.split('.')
        folder_name = track_no[0] + '-chunked'

        path_create = os.path.join(path, folder_name)
        print(path_create)
        folder = os.listdir(path_create)

        for chunk in folder:
            start = time.perf_counter()
            generate_mean(chunk, path_create)
            remaining = 3-(time.perf_counter()-start)
            # print(remaining)
            time.sleep(remaining)