#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:08:23 2022

@author: selene
"""
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from collections import OrderedDict
import threading
import os


# settings
output_dir = r'/Users/irena/Documents/geo/sentinel'         
query_excel_fn = "searchresults.xls"      
success_excel_fn = "dwsuccess.xls"     
aoi_geojson_fp = r"/Users/irena/Documents/geo/sentinel/test.geojson"
query_kwargs = {
    
    "platformname" :'Sentinel-1',
    "date" : ('20180101','20181231'),
    "producttype" : 'SLC',
    "sensoroperationalmode" : 'IW',
    # "orbitnumber" : 16302,
    # "relativeorbitnumber" : 130,
    # "orbitdirection" : 'ASCENDING',
}
thread_num = 3


def products_to_excel(products, fp):
    infos = {
    }
    for item in products:
        info = products[item]
        for name in info:
            value = info[name]
            if name in infos:
                infos[name].append(value)
            else:
                infos[name] = [
                    value
                ]
    dict_to_excel(infos, fp)


def dict_to_excel(d, fp):
    import pandas as pd
    data = [d[key] for key in d]
    df = pd.DataFrame(data=data).T
    df.columns = d.keys()
    df.to_excel(fp)

threads = []
def download_one(api, product, product_info):
    # download
    api.download(product, directory_path=output_dir)
    # save info
    success_products[product] = product_info
    products_to_excel(success_products, success_excel_fp)

    print('\t[SUCCESS] {}/{}'.format(len(success_products), total))
    # del products[product]
    # print('\t[surplus] {}'.format(len(products) ))

# program variable
success_products = OrderedDict()
products = OrderedDict()
if __name__ == '__main__':
    query_excel_fp = os.path.join(output_dir, query_excel_fn)
    success_excel_fp = os.path.join(output_dir, success_excel_fn)

    # 用户名,密码
    api = SentinelAPI('uid', 'password')

    # 搜索
    footprint = geojson_to_wkt( read_geojson(aoi_geojson_fp)) 
    kw = query_kwargs.copy() 
    results = api.query(footprint, **kw)
    products.update(results)

    # ui
    total = len(products)
    print("[Total] {} ".format(total) )
    # save file
    products_to_excel(products, query_excel_fp)

    try:
        cnt = 1
        for product in products:
            product_odata = api.get_product_odata(product)
            if not product_odata['Online']:
                print("[activate prod {}] {}".format(cnt, product_odata['date']))
                api.download(product, output_dir) 
                cnt += 1
    except:
        print("activation invalid")
        pass

    while len(success_products)!=total:
        for product in products:
            product_odata = api.get_product_odata(product)

            if product_odata['Online']: 
                
                print('[Online] {} {}'.format(product_odata['date'], product_odata['title']) )
                # print("thread {}".format(threading.active_count()) ) #debug
                if threading.active_count()<=thread_num: 
                    t = threading.Thread(download_one(api, product, products[product]) )
                    t.start()
            else:
               
                print("[Offine] {}".format(product_odata['date'] ))