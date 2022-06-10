import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd

import utils


tracks = utils.load('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_metadata/tracks.csv')

subset = tracks.index[tracks['set', 'subset'] <= 'medium']
tracks = tracks.loc[subset]

train = tracks.index[tracks['set', 'split'] == 'training'].tolist()
val = tracks.index[tracks['set', 'split'] == 'validation'].tolist()
test = tracks.index[tracks['set', 'split'] == 'test'].tolist()


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
path_create = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_medium/'
# to_create = ['133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154']
save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_medium1/'

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

# 'C:/Users/jeb1618/masters/models/saved_models/model_v2'

# test_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds/swipe-chunked/'
# save_to = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/test_random/'

# create_image('swipe_20.wav', path_create, save_to)

folder_dir = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_medium/000/'
df = pd.read_csv('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/medium_multigenre.csv')
ref = utils.load('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/medium_multigenre.csv')


# img = create_image('000002.mp3', folder_dir)
# classify_image(img, save_to)

def create_and_classify(file_name, path, save_to):

    # print('begin')

    track_no = file_name.split('.')
    track = track_no[0]
    track_id = int(track)

    # print(track, track_id)

    location = os.path.join(path, file_name)

    # print(location)

    data, sr = librosa.load(location, res_type='kaiser_fast')

    spec = librosa.feature.melspectrogram(y=data, sr=sr)
    spec_big = librosa.power_to_db(spec)

    # print("librosa created")

    img = librosa.display.specshow(spec_big)

    # print("img created")

    # index = df[df['track_id'==track_id]].index.values
    # print(index)
    # set_ = df.loc[track_id]['split']
    if track_id in train:
        set_ = 'training'
    elif track_id in test:
        set_ = 'test'
    else:
        set_ = 'validation'

    # print("split found:", set_)

    target_path = os.path.join(save_to, set_)

    plt.savefig(target_path+"/{}.png".format(track), bbox_inches='tight')
    plt.clf()

    # print("ok next")

# create_and_classify('000002.mp3', folder_dir, save_to)

# process_list = ['000', '001', '002']
# process_list = ['018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155']

fma_medium1 = ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024','025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053']
fma_medium2 = ['085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104']
# '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', 
fma_medium3 = ['105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155']


audio_files = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_medium/'
for folder in os.listdir(audio_files):
    # print('trying: folder', folder)
    if folder in fma_medium3:
        print("processing: folder", folder)
        folder_dir = os.path.join(audio_files, folder)
        for track in os.listdir(folder_dir):
            try:
                # if folder in fma_medium1:
                create_and_classify(track, folder_dir, 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_medium3/')
                # elif folder in fma_medium2:
                    # create_and_classify(track, folder_dir, 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_medium2/')
                # else:
                    # create_and_classify(track, folder_dir, 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_medium3/')
            except:
                print("exception track", track)
    else:
        pass

# create_and_classify('025000.mp3', 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_medium/025/', 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_medium1/')

# for i, track in enumerate(df['split']):
#     if i<10:
#         print(track)
#         print(df.index[df['split'==track]])