import json

start_date = '0101'
end_date = '1231'
state = 0
'''
with open('./keyword_正妹_0101_1231.json', 'r') as json_file:
    image1 = json.load(json_file)
with open("C:/Users/Ray/Desktop/Shen/keyword_正妹_0101_1231.json", 'r') as json_file:
    image2 = json.load(json_file)
    
#for json_str in json_list:
#    images1 = json.loads(json_str)

count=0
for i in image2['image_urls']:
    if i not in image1['image_urls']:
        print(i)
        count+=1
print(count)
'''
    #print(i)
#for i in image1:
#    print(i["image_urls"])



import json

with open('./keyword_正妹_0101_1231.json', 'r', encoding="utf-8") as f1, open('C:/Users/Ray/Desktop/Shen/keyword_正妹_0101_1231.json', "r", encoding="utf-8") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)
    print(len(data1["image_urls"]), len(set(data1["image_urls"])))
    print(len(data2["images_urls"]), len(set(data2["images_urls"])))
    print(len(set(data1["image_urls"])) == len(set(data2["images_urls"])))
    for i in (set(data1["image_urls"]) - set(data2["images_urls"]) ):
        print(i)
