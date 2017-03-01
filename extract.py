from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import cPickle as pickle
import json

base_model = VGG16(weights='imagenet')
# note: this is using the pre-softmax layer form the 16 layer VGGNet.
# not sure if Lazaridou et al a0 use 16 or 19 layer VGGNet and b)
# use this exact layer ("second to last fully connected layer" could
# also mean fc1?)
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
cropids = []

with open("boxes.json", "r") as f:
    boxes = json.load(f)
    
for img in boxes.keys():
    for obj in boxes[img]['objects']:
        cropids.append( (img, obj) )

print "nr of pairs: " + str(len(cropids))

features = np.zeros( (len(cropids), 4096) )

for i in range(len(cropids)):
    img_path = "data/images/crops/{}_{}.jpg".format(cropids[i][0], cropids[i][1])
    img = image.load_img(img_path, target_size=(224, 224))
    # just copying these next 2 lines from keras.applications example.
    # why?  because preprocess expects multiple images at a time?
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x)
#    print feats.shape
#    print feats[0:10]
    features[i,:] = feats[0,:]
    
# save feature array
np.save("data/crops/VGG16_fc2.npy", features)
    
# dump list with (img, obj) tuples
# (same order as numpy array)
with open("crop_ids.p", wb) as f:
    pickle.dump(cropids, f)
