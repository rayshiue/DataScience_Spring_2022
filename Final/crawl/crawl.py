from selenium import webdriver
import time
import pandas as pd
from selenium.webdriver.common.by import By
import json
from tqdm import tqdm
import jsonlines

def main():
    target = open("./target.txt", "r")
    for _name in target.readlines():

        name = _name.strip()
        print(name)

        file_count = 0
        try:
            with jsonlines.open("./news_dataset/{}.jsonl".format(name)) as reader:
                for line in reader:
                    file_count += 1
        except Exception as e:
                print("find no {}.jsonl".format(name))
        print("Number of original data:", file_count)

        options = webdriver.ChromeOptions()
        options.add_argument("--lang=en-US")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36")
        options.add_argument('--timeout=180')
        driver = webdriver.Chrome( options=options)
        
        url = 'https://www.youtube.com/@{}/videos'.format(name)
        driver.get(url)
        

        SCROLL_PAUSE_TIME = 1
        time.sleep(3)
        last_height = driver.execute_script("return document.documentElement.scrollTop")
        print("start")
        
        error_flag = False
        index_append = 0
        titles = []
        views = []
        video_lens =[]
        images =[]
        upload_times = []
        data_len = 0

        start_all =time.time()
        while True:
            
            #print("A")
            try:
                driver.execute_script("window.scrollTo(0, arguments[0]);", last_height)
                time.sleep(SCROLL_PAUSE_TIME)
                new_height = driver.execute_script("return document.documentElement.scrollHeight")
            except Exception as e:
                print("\n")
                print(e)
                print("Error: execute_script_A")
                error_flag = True
                break
            
        
            #print("B")
            if last_height == new_height :
                time.sleep(3)
                try:
                    driver.execute_script("window.scrollTo(0, arguments[0]);", last_height)
                except Exception as e:
                    print("\n")
                    print(e)
                    print("Error: execute_script_B")
                    error_flag = True
                    break
                time.sleep(3)

                if last_height == new_height :
                    print("\n")
                    print("break with height")
                    break



            
            #write data
            #print("C")
            try:
                title_info = driver.find_elements(By.ID, "video-title-link")
                image_info = driver.find_elements(By.XPATH,'//*[@id="dismissible"]/ytd-thumbnail/a/yt-image/img')
                video_len_info = driver.find_elements(By.XPATH,'//*[@id="dismissible"]/ytd-thumbnail/a/div/ytd-thumbnail-overlay-time-status-renderer/span')
            except Exception as e:
                print("\n")
                print(e)
                print("Error: find_element")
                error_flag = True
                break

            

            #print("length: {} ".format(len(title_info)))
            #print("index_append: {}".format(index_append))
            #print("D")

            try:
                start = time.time()
                progress = tqdm(total=len(title_info)-index_append, position=0, leave=True)
                for i in range(index_append, len(title_info)):
                    progress.update(1)
                    data = title_info[i].get_attribute("aria-label")

                    if data == None:
                        continue
                    if data.find("ago")>0:
                        split = data.split("ago")
                        upload_time = split[-2].split()[-2:]
                        upload_times.append(upload_time[0] + " " + upload_time[1])
                        view = split[-1].split()[-2].replace(",", "")
                        if view == "No":
                            view = 0
                        else:
                            view = int(view)
                        views.append(view)
                        title = title_info[i].get_attribute("title") 
                        titles.append(title)
                    else:
                        print("\n")
                        print("find no ago : {}".format(data))
                progress.close()
                end = time.time()
                #print("title time :{}".format(end-start))
            except Exception as e:
                print("\n")
                print(e)
                print("Error: title_info")
                error_flag = True
                break

    
            #####
            index_append = len(titles)
            ##### 
            
            #print("E")
            start = time.time()
            sum_i=0
            progress = tqdm(total=len(image_info), position=0, leave=True)
            try:
                for i in range(len(image_info)):
                    progress.update(1)
                    if i < len(images):
                        if images[i] == None:
                            sum_i+=1
                            #images[i] = image_info[i].get_attribute("src")
                            try:
                                images[i] = image_info[i].get_attribute("src")
                            except Exception as e:
                                print("\n")
                                print(e)
                                print("Error: images[i] = image_info[i].get_attribute('src')")
                                print("i = {}".format(i))
                                print("image_info length = {}".format(len(image_info)))
                                images[i] = None
                    else:
                        try:
                            images.append(image_info[i].get_attribute("src"))
                        except Exception as e:
                            print("\n")
                            print(e)
                            print("images.append(image_info[i].get_attribute('src'))")
                            print("i = {}".format(i))
                            print("image_info length = {}".format(len(image_info)))
                            images.append(None)
                progress.close()
                #print("\n")
                #print("sum_i = {}".format(sum_i))
                end = time.time()
                #print("image time :{}".format(end-start))
            except Exception as e:
                print("\n")
                print(e)
                print("Error: image_info")
        

            #print("F")
            
            try:
                start = time.time()
                sum_v=0
                progress = tqdm(total=len(video_len_info), position=0, leave=True)
                for i in range(len(video_len_info)):
                    progress.update(1)
                    if i < len(video_lens):
                        if video_lens[i] == None:
                            sum_v+=1
                            #video_lens[i] = video_len_info[i].text
                            try:
                                video_lens[i] = video_len_info[i].text
                            except:
                                print("\n")
                                print("Error: video_lens[i] = video_len_info[i].text")
                                print("i = {}".format(i))
                                print("video_len_info length = {}".format(video_len_info))
                                video_lens[i] = None
                    else:
                        try:
                            video_lens.append(video_len_info[i].text)  
                        except:
                            print("\n")
                            print("video_lens.append(video_len_info[i].text)  ")
                            print("i = {}".format(i))
                            print("video_len_info length = {}".format(len(video_len_info)))
                            video_lens.append(None)
                progress.close()
                #print("\n")
                #print("sum_v = {}".format(sum_v))
                end = time.time()
                #print("video time :{}".format(end-start))
            except Exception as e:
                print("\n")
                print(e)
                print("Error: video_len_info")

               

            data_len = len(titles)

            if data_len > file_count:
                try:
                    f = open("./news_dataset/{}.jsonl".format(name), "w", encoding='utf-8')
                    for i in range(data_len):
                        title = titles[i]
                        view = views[i]
                        video_len = video_lens[i]
                        image = images[i]
                        upload_time = upload_times[i]
                        temp_dict = {
                            "title" : title,
                            "views" : view,
                            "length" : video_len,
                            "image" : image,
                            "upload_time" : upload_time
                        }
                        json.dump(temp_dict, f, ensure_ascii=False)
                        f.write("\n")
                    f.close()
                except Exception as e:
                    print("\n")
                    print(e)
                    print("title len{} image len{} video len{}".format(len(titles), len(images), len(video_lens)))

            #check time
            #print("G")
            try:
                temp = driver.find_elements(By.ID, "video-title-link")
                temp = temp[-2].get_attribute("aria-label")
                if temp.find("ago")>0:
                    split = temp.split("ago")
                    upload_time = split[-2].split()[-2:]
                    #print(new_height, (upload_time[0] + " " + upload_time[1]), end="            \r")
                    print(new_height, (upload_time[0] + " " + upload_time[1]))
                    if upload_time[1] == "year":
                        break
                
                last_height = new_height
            except Exception as e:
                print("\n")
                print(e)
                print("Error: check")
                error_flag = True
                break
            

        driver.quit()
        if error_flag:
            print("channel {} is out of memory, video count is {}".format(name, data_len))
        end_all =time.time()
        print("time of {} :{}".format(name, end_all-start_all))


# f = open("data.jsonl", "w")
if __name__ == "__main__":
    main()
