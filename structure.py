from refer import REFER
from collections import defaultdict
import json

# missing key trick for nested dict copied from
# http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

data_root = "data"
dataset = "refcoco+"
splitBy = "unc"
refer = REFER(data_root, dataset, splitBy)

boxes = Vividict()
refs = refer.imgToRefs
anns = refer.imgToAnns
for img in refs.keys():
    print img, 
    # general properties of the image
    boxes[img]["filename"] = refs[img][0]["file_name"]
    boxes[img]["split"] = refs[img][0]["split"]
    # bounding boxes with sentences, by annotation id
    #  this only works if/because id in anns == ann_id in refs
    for obj in refs[img]:
        boxes[img]["objects"][obj["ann_id"]]["sentences"] = obj["sentences"]
    for obj in anns[img]:
        if obj["id"] in boxes[img]["objects"].keys():
            boxes[img]["objects"][obj["id"]]["bbox"] = obj["bbox"]
print "Boxes made"
with open("boxes.json","w") as f:
        json.dump(boxes, f)
print "written to boxes.json"
