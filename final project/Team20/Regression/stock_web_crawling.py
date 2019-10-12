# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:41:52 2018

@author: 羅宇呈
"""

import requests
import urllib
import time
import os
import datetime
import urllib.request
import csv
import random




stocks = {
          'MXIC':2337,
          'ASUS':2357,
          'ACER':2353,
          'MSI':2377,
          'AU':2409,
          'HTC':2498,
          'TSMC':2330,
          'Realtek':2379,
          'FPG':1301,
          'Sinosteel':2002
          }


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/49.0.2')]



for company in stocks:

    os.makedirs('./'+company,exist_ok=True)
    
    date=20060101
    
    total_data=[]
    
    while date !=20190101:

        unknown_para=random.random()
        payload = {'response': 'json', 'date':date,'stockNo':stocks[company],'_':unknown_para}

    
        head={'Accept': 'application/json, text/javascript, /; q=0.01',
          'Accept-Encoding': 'gzip, deflate',
          'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
          'Connection': 'keep-alive',
          'Host': 'www.twse.com.tw',
          'Referer': 'http://www.twse.com.tw/zh/page/trading/exchange/STOCK_DAY.html',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36',
          'X-Requested-With': 'XMLHttpRequest'}
    
        req = requests.get("http://www.twse.com.tw/exchangeReport/STOCK_DAY.html",headers=head,params=payload)
    
        if(req.status_code==200):
    
    #http://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=20181213&stockNo=2002&_=1544697988733
    #print(req.url)

            raw_json=req.json()
        
            header=raw_json['fields']
        
            total_data=total_data+raw_json['data']

        #print(total_data)
                
        #print(str(date)[4],str(date)[5])
        
        #print(type(str(date)[4]),type(1))
        
        
            if(str(date)[4]==str(1) and str(date)[5]==str(2)):
                date=int((int(date/10000)+1)*10000+101)
                print(date)
            else:
                date=date+100
            
            print(date)
        
            time.sleep(3)
        
        
        
    #print(raw_json['fields'])
    

    with open(company+'.csv', mode='w',newline='') as csvfile:
        
         #employee_writer = csv.DictWriter(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,fieldnames=raw_json['fields'])
         #employee_writer.writeheader()
         writer = csv.writer(csvfile)
         writer.writerow(header)
         
         for line in total_data:
             writer.writerow(line)     
    
    #print(req.apparent_encoding)

    #print(req.json())
        
    #print(req_content)