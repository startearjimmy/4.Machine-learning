import requests
import smtplib
import urllib.parse
import pandas as pd
from requests_html import HTML
from multiprocessing import Pool
import time

def fetch(url):
    #response = requests.get(url)
    response = requests.get(url)
    html = HTML(html = response.text)
    return html

if __name__ == '__main__':
    start = time.time()
    # cols
    taipei_weather = list()
    # taipei_weather = {
    #     'date':[],
    #     'temperature':[],
    #     'weather':[]
    # }
    # date = list()
    # temperature = list()
    # weather = list()
    #crawler
    url_pre = 'http://www.tianqihoubao.com/lishi/taibei/month/'
    index = '201104.html'
    while index != '201812.html':
        url = urllib.parse.urljoin(url_pre, index)
        print(url)
        #get data
        html = fetch(url)
        rows = html.find('tr')
        rows = rows[1:]
        for row in rows:
            cols = row.find('td')
            # taipei_weather['date'].append(cols[0].text)
            # taipei_weather['weather'].append(cols[1].text)
            # taipei_weather['temperature'].append(cols[2].text)
            taipei_weather.append((cols[0].text,cols[1].text,cols[2].text))
        #index add
        index_tmp = 0
        if index[5] == '9':
            tmp = int(index[4]) +1
            index = index[:4] + str(tmp) + '0' + index[6:]
        elif index[5] == '2' and index[4] =='1':
            tmp = int(index[3]) +1
            index = index[:3] + str(tmp) + '0'+'1' + index[6:]
        else:
            tmp = int(index[5]) +1
            index = index[:5] + str(tmp) + index[6:]
    df = pd.DataFrame.from_records(taipei_weather,columns = ['date','weather','temperature'])
    df.to_csv('taipei_weather.csv',encoding='utf_8')