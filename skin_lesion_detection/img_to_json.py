import os
import glob
import json
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

def get_image(path):
    img = image.load_img(path, target_size=(150, 150, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.ascontiguousarray(preprocess_input(x))
    return x


Blob_folder = "Blobs"
# output_folder = "json_files"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
fnames = glob.glob(Blob_folder + "/*.png")
fnames.sort()
fnames = fnames[:5]
for i,fname in enumerate(fnames):
    img = get_image(fname)
    img.tolist()
    json.dump(img.tolist(), open(str(i).zfill(3) + ".json", "w"))
