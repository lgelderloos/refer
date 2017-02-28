from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

base_model = VGG16(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)



feats = numpy.zeros(len(objects), 4096)
for all objects (you can probably use refer-code for this):
    img_path = # path to crop.jpg
    img = image.load_img(img_path, target_size=(224, 224))
    # just copying these next 2 lines from keras.applications example.
    # try figuring out why you need it
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    # print features.shape
    # change numpy array at index to features
