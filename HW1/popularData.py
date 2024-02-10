
import argparse
from bs4 import BeautifulSoup
import requests
import time
import json
import re
start = time.time()

front = '[Hh][Tt][Tt][Pp][Ss]{0,1}://.+.'
pat = front + '[Jj][Pp][Gg]\n|'+ front+'[Gg][Ii][Ff]\n|' + front+'[Pp][Nn][Gg]\n|' + front+'[Jj][Pp][Ee][Gg]\n'

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
    all_article_file = open('./all_article.jsonl', 'w')
    all_popular_file = open('./all_popular.jsonl', 'w')
    action = 1
elif args.action == 'push':
    start_date = args.arg1
    end_date = args.arg2
    push_file = open('./push_'+start_date+'_'+end_date+'.json', 'w')
    action = 2
elif args.action == 'popular':
    start_date = args.arg1
    end_date = args.arg2
    popular_file = open('./1.txt', 'w')
    non_popular_file = open('./0.txt', 'w')
    action = 3
elif args.action == 'keyword':
    keyword = args.arg1
    start_date = args.arg2
    end_date = args.arg3
    keyword_file = open('./keyword_'+keyword+'_'+start_date+'_'+end_date+'.json', 'w')
    popular_file = open('./train_data_nocom/1.txt', 'w')
    non_popular_file = open('./train_data_nocom/0.txt', 'w')
    action = 4

userlist_like = {}
userlist_boo = {}
keyword_imgs = {}
popular = {}
img_list = []
popular_img_list = []
non_popular_img_list = []
num_popular = 0

state = 0
all_like = 0
all_boo = 0
def parse_date(date):
    if date[0]=='0':
        date = ' '+date[1]+'/'+date[2:]
    else:
        date = date[0:2]+'/'+date[2:]
    return date
start_date = parse_date(start_date)
end_date = parse_date(end_date)
print(start_date)
print(end_date)

r = requests.Session()
payload = {'from': '/bbs/Beauty/index.html', 'yes': 'yes'}
url = "https://www.ptt.cc/bbs/Beauty/index.html"
r1 = r.post("https://www.ptt.cc/ask/over18?from=%2Fbbs%2FBeauty%2Findex.html", payload)
flag = True
while True:

    r2 = r.get(url)
    content = r2.text
    soup = BeautifulSoup(r2.text, 'html.parser')
    u = soup.select("div.btn-group.btn-group-paging a")#上一頁按鈕的a標籤
    url = "https://www.ptt.cc"+ u[1]["href"] #組合出上一頁的網址
    all_titles = soup.find_all(class_ = "r-ent")
    all_titles.reverse()
    for t in all_titles:
        #title = str(t.find(class_ = "title").find('a').string)
        title = t.find(class_ = "title").find('a')
        date = str(t.find(class_ = "date").string)
        push = t.find(class_ = "nrec").find('span')
        if flag and date[2:]=='01':
            print(date)
            flag = False
        elif date[2:]!='01':
            flag = True
 
        if title!=None:
            title = str(title.string)
            if title[0:4] == '[公告]' or title[0:8] == 'Fw: [公告]':
                continue
            if date!='12/31' and state == 0:
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
        url_link = t.find(class_ = "title").find('a').get('href')
        #print(title)
        #print(url_link)

        link = 'https://www.ptt.cc' + url_link
        if action==1:
            article = {"date": date, "title": title, "url": link}
            json.dump(article, all_article_file)
            all_article_file.write('\n')  

            if push!=None:
                if push.string=='爆':
                    json.dump(article, all_popular_file)
                    all_popular_file.write('\n') 
            continue
        if action==3 or action==4:
            if push==None:
                isPopular = False
            elif push.string=='爆':
                isPopular = True
            elif push.string[0]=='X':
                isPopular = False
            elif int(push.string)<=35:
                isPopular = False
            else:
                isPopular = True
        

        r2 = r.get(link)
        content = r2.text
        soup2 = BeautifulSoup(content, 'html.parser')

        if action == 3:
            num_popular+=1
            imgs = soup2.findAll('a')
            for img in imgs:
                result = re.findall(pat, str(img['href']))
                if result!=[]:
                    if isPopular:
                        popular_img_list.append(result[0])
                    else:
                        non_popular_img_list.append(result[0])
            continue
        if action==4:
            metaval = soup2.find_all(class_ = "article-meta-value")
            metatag = soup2.find_all(class_ = "article-meta-tag")
            main_container = soup2.find(id = 'main-container')
            if main_container!=None:
                idx = main_container.text.find('※ 發信站')
                if idx==-1:
                    continue
                else:
                    imgs = re.findall(pat, str(main_container.text)[:idx])
                    for img in imgs:
                        if isPopular:
                            popular_img_list.append(img)
                        else:
                            non_popular_img_list.append(img)
            continue

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
    for img in popular_img_list:
        popular_file.write(img+'\n')
    for img in non_popular_img_list:
        non_popular_file.write(img+'\n')
    popular_file.close()
    non_popular_file.close()

if action==4:
    popular["number_of_popular_articles"] = num_popular
    for img in popular_img_list:
        popular_file.write(img+'\n')
    for img in non_popular_img_list:
        non_popular_file.write(img+'\n')
    popular_file.close()
    non_popular_file.close()

end = time.time()
print('total spending time = {:.2f}'.format((end - start)/60))