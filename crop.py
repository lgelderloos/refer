import json
from random import sample
from skimage import io

##########################################################

def remove_id(filename):
    filename = filename.rsplit('_',1)[0]
    filename = filename + ".jpg"
    return filename

def actual_bbox(bbox):
    actual = []
    for f in bbox:
        actual.append(int(round(f)))
    return actual

def crop(img, bbox):
    bbox = actual_bbox(bbox)
    x_end = bbox[0] + bbox[2]
    y_end = bbox[1] + bbox[3]
    #print "x_end:", x_end, "y_end:", y_end
    #print "img shape: ", img.shape
    #print "bbox:", bbox
    if img.ndim == 3:
        crop = img[bbox[1]:y_end,bbox[0]:x_end,:]
    elif img.ndim ==2:
        crop = img[bbox[1]:y_end,bbox[0]:x_end]
    return crop

def load_pic(filename):
    if "test2014" in filename:
        data = "data/images/mscoco/images/test2014/"
    elif "train2014" in filename:
        data = "data/images/mscoco/images/train2014/"
    # should do this properly
    else:
        print "cannot find {}; directory unknown".format(filename)
        return
    image = io.imread((data+remove_id(filename)))
    return image

def save_crop(img, img_id, obj, obj_id):
    boxcrop = crop(img, obj['bbox']) 
    #print "shape crop: ", boxcrop.shape
    io.imsave("data/images/crops/{}_{}.jpg".format(img_id, obj_id), boxcrop)
    
#########################################################

with open("boxes.json", "r") as f:
    boxes = json.load(f)
    
for img in boxes.keys():
    print img, 
    pic = load_pic(boxes[img]['filename'])
    print pic.shape, 
    for obj in boxes[img]['objects']:
            #print "object: ", obj
            #print boxes[img]['objects'][obj]['sentences']
            save_crop(pic, img,  boxes[img]['objects'][obj], obj)
