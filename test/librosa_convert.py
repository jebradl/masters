import librosa
import librosa.display
import time
import matplotlib.pyplot as plt
import numpy as np

music = 'c:/Users/night/Documents/09/school/actual-masters/git/masters/test/sample_sounds/next_level.mp3'

start = time.perf_counter()

data, sr = librosa.load(music, res_type='kaiser_fast') # default kaiser_best

print(time.perf_counter()-start)
print(data.shape, sr)

spec = librosa.feature.melspectrogram(y=data, sr=sr)
print(spec)
# librosa.display.specshow(spec,y_axis='mel', x_axis='s', sr=sr)
# plt.colorbar()
# plt.show()

# db_spec = librosa.power_to_db(spec, ref=np.max,)
# librosa.display.specshow(db_spec,y_axis='mel', x_axis='s', sr=sr)
# plt.colorbar();
# plt.show()

data_h, data_p = librosa.effects.hpss(data)
spec_h = librosa.feature.melspectrogram(data_h)
spec_p = librosa.feature.melspectrogram(data_p)
db_spec_h = librosa.power_to_db(spec_h,ref=np.max)
db_spec_p = librosa.power_to_db(spec_p,ref=np.max)

# librosa.display.specshow(db_spec_p,y_axis='mel', x_axis='s', sr=sr)
# plt.colorbar();
# plt.show()