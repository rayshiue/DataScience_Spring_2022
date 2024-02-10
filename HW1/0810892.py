
import argparse
from bs4 import BeautifulSoup
import requests
import time
import json
import re
start = time.time()

front = '[Hh][Tt][Tt][Pp][Ss]{0,1}://.+'
pat = front + '\.[Jj][Pp][Gg]$|'+ front+'\.[Gg][Ii][Ff]$|' + front+'\.[Pp][Nn][Gg]$|' + front+'\.[Jj][Pp][Ee][Gg]$'

parser = argparse.ArgumentParser()
parser.add_argument("action")
parser.add_argument("arg1",nargs='?')
parser.add_argument("arg2",nargs='?')
parser.add_argument("arg3",nargs='?')
args = parser.parse_args()
action = 0
if args.action == 'crawl':
    start_date = '0101'
    end_date = '1231'
    all_article_file = open('./all_article.jsonl', 'w', encoding="utf-8")
    all_popular_file = open('./all_popular.jsonl', 'w', encoding="utf-8")
    action = 1
elif args.action == 'push':
    start_date = args.arg1
    end_date = args.arg2
    push_file = open('./push_'+start_date+'_'+end_date+'.json', 'w', encoding="utf-8")
    action = 2
elif args.action == 'popular':
    start_date = args.arg1
    end_date = args.arg2
    popular_file = open('./popular_'+start_date+'_'+end_date+'.json', 'w', encoding="utf-8")
    action = 3
elif args.action == 'keyword':
    keyword = args.arg1
    start_date = args.arg2
    end_date = args.arg3
    keyword_file = open('./keyword_'+keyword+'_'+start_date+'_'+end_date+'.json', 'w', encoding="utf-8")
    action = 4

userlist_like = {}
userlist_boo = {}
keyword_imgs = {}
popular = {}
img_list = []
num_popular = 0
flag = True

state = 0
all_like = 0
all_boo = 0
def parse_date(date):
    if date[0]=='0':
        date = ' '+date[1]+'/'+date[2:]
    else:
        date = date[0:2]+'/'+date[2:]
    return date

r = requests.Session()
payload = {'from': '/bbs/Beauty/index.html', 'yes': 'yes'}
url = "https://www.ptt.cc/bbs/Beauty/index.html"
r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FBeauty%2Findex.html", payload)

print('Waiting...')

if action==2 or action==4:
    with open('./all_article.jsonl', 'r', encoding="utf-8") as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        date = result['date']

        if date == end_date and state==0:
            state = 1
        elif date==start_date and state==1:
            state=2
        elif date!=start_date and state==2:
            break
        if state==0:
            continue

        if flag and date[2:]=='01':
            print(date)
            flag = False
        elif date[2:]!='01':
            flag = True

        r2 = r.get(result['url'])
        content = r2.text
        soup2 = BeautifulSoup(content, 'html.parser')
        if action == 2:
            allcomment = soup2.find_all(class_ = "push")
            for comment in allcomment:
                down = comment.find(class_ = "f1 hl push-tag")
                up = comment.find(class_ = "hl push-tag")
                if down!=None:
                    if str(down.string)[0] == "噓":
                        userid = str(comment.find(class_ = "f3 hl push-userid").string)
                        if userid not in userlist_boo:
                            userlist_boo[userid] = 1
                        else:
                            userlist_boo[userid] += 1
                        all_boo+=1
                if up!=None:
                    if str(up.string)[0] == "推":
                        userid = str(comment.find(class_ = "f3 hl push-userid").string)
                        if userid not in userlist_like:
                            userlist_like[userid] = 1
                        else:
                            userlist_like[userid] += 1
                        all_like+=1
        if action==4:
            main_container = soup2.find(id = 'main-container')
            if main_container!=None:
                idx = main_container.text.find('※ 發信站')
                if idx==-1:
                    continue
                else:
                    if keyword in main_container.text[:idx]:
                        imgs = soup2.findAll('a')
                        for img in imgs:
                            result = re.findall(pat, str(img['href']))
                            if result!=[]:
                                img_list.append(result[0])
        time.sleep(0.01)

start_date = parse_date(start_date)
end_date = parse_date(end_date)

while action==1 or action==3:

    r2 = r.get(url)
    content = r2.text
    soup = BeautifulSoup(r2.text, 'html.parser')
    u = soup.select("div.btn-group.btn-group-paging a")
    url = "https://www.ptt.cc"+ u[1]["href"]
    all_titles = soup.find_all(class_ = "r-ent")
    all_titles.reverse()

    for t in all_titles:
        title = t.find(class_ = "title").find('a')
        date = str(t.find(class_ = "date").string)
        push = t.find(class_ = "nrec").find('span')
        if title!=None:
            title = str(title.string)
            if title[0:4] == '[公告]' or title[0:8] == 'Fw: [公告]':
                continue
            if date=='12/31' and state == 0:
                state = 1
            if state==1 and date==end_date:
                state = 2
            elif state==2 and date==start_date:
                state = 3
            elif state==3 and date!=start_date:
                state = 4
                break
            if state==0 or state==1:
                continue
        else:
            continue
        if flag and date[3:]=='01':
            print(date)
            flag = False
        elif date[3:]!='01':
            flag = True

        url_link = t.find(class_ = "title").find('a').get('href')
        link = 'https://www.ptt.cc' + url_link
        if action==1:
            if date[0] == ' ':
                date = '0'+date[1]+date[3:]
            else:
                date = date[0:2] + date[3:]

            article = {"date": date, "title": title, "url": link}
            json.dump(article, all_article_file, ensure_ascii=False)
            all_article_file.write('\n')

            if push!=None:
                if push.string=='爆':
                    json.dump(article, all_popular_file, ensure_ascii=False)
                    all_popular_file.write('\n')
            continue
        if action==3:
            if push==None:
                continue
            elif push.string!='爆':
                continue
        
        r2 = r.get(link)
        content = r2.text
        soup2 = BeautifulSoup(content, 'html.parser')
        if action == 3:
            num_popular+=1
            imgs = soup2.findAll('a')
            for img in imgs:
                result = re.findall(pat, str(img['href']))
                if result!=[]:
                    img_list.append(result[0])
            continue
        time.sleep(0.01)
    if state==4:
        break
    time.sleep(0.01)

if action==1:
    all_article_file.close() 
    all_popular_file.close()

if action==2:
    sorted_like = sorted(userlist_like.items(), key=lambda x: (x[1], x[0]), reverse=True)
    sorted_boo = sorted(userlist_boo.items(), key=lambda x: (x[1], x[0]), reverse=True)
    output = {}
    output["all_like"] = all_like
    output["all_boo"] = all_boo
    for idx, user in enumerate(sorted_like):
        if idx>=10:
            break
        output["like {}".format(idx+1)] = {"user_id": user[0], "count": user[1]}
    for idx, user in enumerate(sorted_boo):
        if idx>=10:
            break
        output["boo {}".format(idx+1)] = {"user_id": user[0], "count": user[1]}
    push_file.write('{')
    for key in output:
        push_file.write('\n')
        s = '"' + str(key) + '": ' + str(output[key]).replace("\'", "\"")
        if key!="boo 10":
            s+=","
        push_file.writelines(s)
    push_file.write('\n'+'}')
    push_file.close()

if action==3:
    popular["number_of_popular_articles"] = num_popular
    popular["image_urls"] = img_list
    json.dump(popular, popular_file, indent = 0, ensure_ascii=False)
    popular_file.close()

if action==4:
    keyword_imgs["image_urls"] = img_list
    json.dump(keyword_imgs, keyword_file, indent = 0, ensure_ascii=False)
    keyword_file.close()

end = time.time()
print('total spending time = {:.2f} mins'.format((end - start)/60))