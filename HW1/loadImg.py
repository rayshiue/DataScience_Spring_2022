import urllib.request

f = open('./train_data_nocom/1.txt', 'r')

for i, line in enumerate(f.readlines()):
    if i<9224*2:
        continue
    if i%2==1:
        continue
    print(i)
    try: urllib.request.urlretrieve(line,"./train_data_nocom/1/{}.png".format(int(i/2)))
    except:
        continue
f.close()

f = open('./train_data_nocom/0.txt', 'r')
for i, line in enumerate(f.readlines()):
    print(i)
    try:  urllib.request.urlretrieve(line,"./train_data_nocom/0/{}.png".format(int(i/2)))
    except: 
        continue
f.close()
