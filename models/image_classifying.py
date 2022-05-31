## ok so we need to move images into folders for their respective genres i believe
## so we have to look at the dataset, find all the genres we're using, create a folder for each
## then iterate through the images and move them to their folder respectively 

import os
import utils
import shutil
from tqdm import tqdm

tracks = utils.load('c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/fma_metadata/tracks.csv')
folders_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/classified_small/'
images_path = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/models/data/fma/img/'
images = os.listdir(images_path)

small = tracks[tracks['set', 'subset'] <= 'small']
# print(small.shape)
# print('{} top-level genres'.format(len(small['track', 'genre_top'].unique())))
# print(small['track', 'genre_top'].unique())

# genres_list = (small['track', 'genre_top'].unique()).tolist()
# print(genres_list)
# for genre in genres_list:
#     path_create = os.path.join(folders_path, genre)
#     print(path_create)
#     os.makedirs(path_create)
#     # folder = os.listdir(path_create)

def classify_image(image):
    # so get an image and take just the number
    # get what genre it is from the csv
    # move the image from the img folder to the correct folder

    track_no = image.split('.')
    track = int(track_no[0])

    genre = small.loc[track]['track', 'genre_top']
    # print('track {} is a {} track'.format(track, genre))

    target_path = os.path.join(folders_path, genre)
    # print(target_path)

    shutil.copyfile(images_path+'/'+image, target_path+'/'+image)


# classify_image('000002.png')


for image in tqdm(images):
    classify_image(image)
