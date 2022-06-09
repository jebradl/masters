import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd


def create_image(file_name, path):
    track_no = file_name.split('.')
    track = track_no[0]

    location = os.path.join(path, file_name)

    data, sr = librosa.load(location, res_type='kaiser_fast')

    spec = librosa.feature.melspectrogram(y=data, sr=sr)
    spec_big = librosa.power_to_db(spec)

    img = librosa.display.specshow(spec_big)

    return img


def classify_image(image, save_to):
    # so get an image and take just the number
    # get what genre it is from the csv
    # move the image from the img folder to the correct folder

    track_no = image.split('.')
    track = int(track_no[0])

    df = pd.read_csv('C:/Users/jeb1618/masters/models/data/fma/medium_multigenre.csv')

    # genre = small.loc[track]['track', 'genre_top']
    set = df.loc[track]['split']
    # print('track {} is a {} track'.format(track, genre))

    target_path = os.path.join(save_to, set)
    # print(target_path)

    plt.savefig(save_to+"{}.png".format(track), bbox_inches='tight')
    plt.clf()

    # shutil.copyfile(images_path+'/'+image, target_path+'/'+image)

# save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/img/'
path_create = 'C:/Users/jeb1618/masters/models/data/fma/fma_medium/'
# to_create = ['133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154']
save_to = 'C:/Users/jeb1618/masters/models/data/fma/classified_medium/'

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

'C:/Users/jeb1618/masters/models/saved_models/model_v2'

# test_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds/swipe-chunked/'
# save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/test_random/'

audio_files = 'C:/Users/jeb1618/masters/models/data/fma/fma_medium/'
# for folder in os.listdir(audio_files):
#     print("processing: folder", folder)
#     folder_dir = os.path.join(audio_files, folder)
#     for track in os.listdir(folder_dir):
#         try:
#             img = create_image(track, folder_dir)
#             classify_image(img, save_to)
#         except:
#             print("exception track", track)

# create_image('swipe_20.wav', path_create, save_to)

folder_dir = 'C:/Users/jeb1618/masters/models/data/fma/fma_medium/000/'



img = create_image('000002.mp3', folder_dir)
# classify_image(img, save_to)
