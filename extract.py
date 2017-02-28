from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import cPickle as pickle

base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
cropids = []

with open("boxes.json", "r") as f:
    boxes = json.load(f)
    
for img in boxes.keys():
    for obj in boxes[img]['objects']:
        cropids.append((img, obj))
print len(cropids)
"""
features = numpy.zeros(len(cropnames), 4096)
for i in range(len(cropids)):
    img_path = "data/images/crops/{}_{}.jpg".format(cropids[i][0], cropids[i][1])
    img = image.load_img(img_path, target_size=(224, 224))
    # just copying these next 2 lines from keras.applications example.
    # why?  because preprocess expects multiple images at a time?
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features[i,:] = model.predict(x)
    
# save feature array
np.save("data/crops/VGGfeats.npy", feats)
    
# dump list with (img, obj) tuples
# (same order as numpy array)
with open("crop_ids.p", wb) as f:
    pickle.dump 
"""
