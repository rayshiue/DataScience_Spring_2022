#shen = "C:/Users/Ray/Desktop/Shen/all_article.jsonl"
#shiue = "C:/Users/Ray/Desktop/all_article.jsonl"

import json

shen = []
hsues = []

with open("C:/Users/Ray/Desktop/Shen/all_article.jsonl", 'r', encoding="utf-8") as f1, open("C:/Users/Ray/Desktop/all_article.jsonl", 'r', encoding="utf-8") as f2:
    #with open('./data/my_filename.jsonl', 'r') as json_file:
    f1 = list(f1)
    f2 = list(f2)
    for json_str in f1:
        result = json.loads(json_str)
        shen.append(result['url'])

    for json_str in f2:
        result = json.loads(json_str)
        hsues.append(result['url'])
        
    #data1 = json.load(f1)
    #data2 = json.load(f2)
    #print(len(data1["image_urls"]), len(set(data1["image_urls"])))
    #print(len(data2["image_urls"]), len(set(data2["image_urls"])))
    #print(len(set(data1["image_urls"])) == len(set(data2["image_urls"])))
    #for i in (set(data2["image_urls"]) - set(data1["image_urls"])):
    #    print(i)
    #print(len(shen))
    #print(len(hsues))
    for i in (set(shen) - set(hsues)):
        print(i)
    #for i in (set(hsues) - set(shen)):
    #    print(i)
        