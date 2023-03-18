data = [{"supercategory": "none", "id": 9, "name": "train"}, 
 {"supercategory": "none", "id": 10, "name": "car"}, 
 {"supercategory": "none", "id": 12, "name": "cat"}, 
 {"supercategory": "none", "id": 13, "name": "chair"}, 
 {"supercategory": "none", "id": 3, "name": "person"}, 
 {"supercategory": "none", "id": 19, "name": "diningtable"}, 
 {"supercategory": "none", "id": 5, "name": "sofa"}, 
 {"supercategory": "none", "id": 4, "name": "horse"}, 
 {"supercategory": "none", "id": 6, "name": "bicycle"}, 
 {"supercategory": "none", "id": 11, "name": "bird"}, 
 {"supercategory": "none", "id": 7, "name": "cow"}, 
 {"supercategory": "none", "id": 16, "name": "aeroplane"}, 
 {"supercategory": "none", "id": 20, "name": "tvmonitor"}, 
 {"supercategory": "none", "id": 17, "name": "bottle"}, 
 {"supercategory": "none", "id": 14, "name": "pottedplant"}, 
 {"supercategory": "none", "id": 8, "name": "boat"}, 
 {"supercategory": "none", "id": 15, "name": "sheep"}, 
 {"supercategory": "none", "id": 18, "name": "bus"}, 
 {"supercategory": "none", "id": 1, "name": "motorbike"}, 
 {"supercategory": "none", "id": 2, "name": "dog"}]

order_dict = dict()
for item in data:
    order_dict[item["id"]] = item["name"]

print(order_dict)

myKeys = list(order_dict.keys())
myKeys.sort()
sorted_name = [order_dict[i] for i in myKeys]
print(myKeys, sorted_name)
