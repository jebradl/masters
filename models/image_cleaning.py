import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time


def create_images(file_name, path, save_to):
    track_no = file_name.split('.')
    track = track_no[0]
    # print(track_no, track)
    
    # print("function: start")
    # print(path)
    location = os.path.join(path, file_name)
    # print("location set")
    # start1 = time.perf_counter()
    data, sr = librosa.load(location, res_type='kaiser_fast')
    # end1 = time.perf_counter()-start1
    # start2 = time.perf_counter()
    # print("librosa formulated")
    spec = librosa.feature.melspectrogram(y=data, sr=sr)
    spec_big = librosa.power_to_db(spec)
    # print(spec_big.shape)
    img = librosa.display.specshow(spec_big)
    # print("image created")
    # end2 = time.perf_counter()-start2
    start3 = time.perf_counter()
    plt.savefig(save_to+"{}.png".format(track), bbox_inches='tight')
    plt.clf()
    end3 = time.perf_counter()-start3

    print("track {} - save time: {:.3f}s".format(track, end3))



# save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/img/'
path_create = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_small/'
to_create = ['133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154']


# for folder in to_create:
#     print('new folder: {}'.format(folder))
#     folder = folder + '/'
#     path = os.path.join(path_create, folder)
#     # print(path)
#     audio_files = os.listdir(path)
#     for file in audio_files:
#         create_images(file, path, save_to)

# file = '042243.mp3'
# create_images(file, path, save_to)

test_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds/swipe-chunked/'
save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/test_random/'

create_images('swipe_20.wav', test_path, save_to)