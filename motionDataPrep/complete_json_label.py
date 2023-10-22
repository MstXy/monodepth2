import json
import os


MYDIR = "/home/admin/Downloads/ljx/labelme_kitti_movementsSegData"
SEQ = "21_2"

all_class = []
json_lst = []
png_lst = []
for f in os.listdir(os.path.join(MYDIR, SEQ)):
    if f.endswith(".json"):
        json_lst.append(os.path.join(f))
    elif f.endswith(".png"):
        png_lst.append(f.strip(".png"))

for fn in json_lst:
    change_flag = False
    f = open(os.path.join(MYDIR, SEQ, fn))
    data = json.load(f)

    for i,shape in enumerate(data["shapes"]):
        cls = int(shape["label"])
        points = shape["points"]

        ## check if points are only one or two points.
        if len(points) < 3:
            print(f)

        #############################
        ## we don't have cls 8 for 02/04
        if SEQ in ["02_2", "04_2"] and cls > 8:
            change_flag = True
            cls = cls - 1
            data["shapes"][i]["label"] = str(cls)    
        
        ## we don't have cls 6,7,8 for 06
        if SEQ in ["06_2"] and cls == 9:
            change_flag = True
            cls = 6
            data["shapes"][i]["label"] = str(cls) 

        ## we have 99, should be 30 for 08
        if SEQ in ["08_2"] and cls == 99:
            change_flag = True
            cls = 30
            data["shapes"][i]["label"] = str(cls) 
        
        ## we don't have cls 2 for 10
        if SEQ in ["10_2"] and cls > 2:
            change_flag = True
            cls = cls - 1
            data["shapes"][i]["label"] = str(cls)    
        
        ## we don't have cls 61-64,70-76,78-87 for 21
        if SEQ in ["21_2"] and cls > 60:
            change_flag = True
            MAP = {65:61, 66:62, 67:63, 68:64, 69:65, 77:66, 88:67}
            cls = MAP[cls]
            data["shapes"][i]["label"] = str(cls)   
        #############################

        if cls not in all_class:
            all_class.append(cls)

    # change file, (replace)
    if change_flag:
        with open(os.path.join(MYDIR, SEQ, fn), "w") as jsonFile:
            json.dump(data, jsonFile, ensure_ascii=False, indent=4)

all_class.sort()
print(all_class)
assert len(all_class) == all_class[-1]

for fn in png_lst:
    if fn + ".json" not in json_lst:
        print(fn)
        # need to write empty json file
        with open(os.path.join(MYDIR, SEQ, fn + '.json'), 'w', encoding='utf-8') as f:
            data = {
                    "version": "5.2.0.post4",
                    "flags": {},
                    "shapes": [],
                    "imagePath": fn + ".png",
                    "imageData": None,
                    "imageHeight": 376,
                    "imageWidth": 1241
                    }
            json.dump(data, f, ensure_ascii=False, indent=4)