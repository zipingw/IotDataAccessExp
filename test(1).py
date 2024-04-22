import requests
import socket
import csv
from math import radians, cos, sin, asin, sqrt
import geoip2.database

# 定义你的域名列表
# 定义要测试的域名列表
domains = ['baidu.com', 'alibaba.com', 'tencent.com', 'weibo.com', 'jd.com',
           '163.com', 'weixin.qq.com', 'v.qq.com', 'iqiyi.com', 'bytedance.com',
           'feishu.cn', 'douyin.com', 'huawei.com', 'csdn.net', 'zhihu.com',
           'naver.com', 'rakuten.co.jp', 'flipkart.com', 'tokopedia.com', 'kooora.com',
           'souq.com', 'yandex.ru',
           'bbc.co.uk', 'leparisien.fr', 'spiegel.de', 'repubblica.it', 'marca.com',
           'svd.se', 'vg.no', 'wp.pl', 'rte.ie', 'nrc.nl',
           'standardmedia.co.ke', 'news24.com', 'vanguardngr.com', 'youm7.com', 'allafrica.com',
           'tunisienumerique.com', 'le360.ma', 'dailymaverick.co.za', 'ethiopianreporter.com', 'monitor.co.ug',
           "google.com",
           "amazon.com",
           "facebook.com",
           "nasa.gov",
           "mit.edu",
           "cbc.ca",
           "uwaterloo.ca",
           "shopify.com",
           "unam.mx",
           "eluniversal.com.mx",
           "globo.com",
           "uol.com.br",
           "mercadolivre.com.br",
           "clarin.com",
           "lanacion.com.ar",
           "emol.com",
           "uc.cl",
           "eltiempo.com",
           "unal.edu.co",
           "rpp.pe",
           "abc.net.au",
           "unsw.edu.au",
           "csiro.au",
           "anz.com",
           "woolworths.com.au",
           "westpac.com.au",
           "uq.edu.au",
           "monash.edu",
           "commbank.com.au",
           "telstra.com.au"
           ]
ips = ['110.242.68.66',
        '59.82.122.231',
        '109.244.194.121',
        '106.63.15.10',
        '106.39.171.134',
        '123.58.180.8',
        '101.227.132.126',
        '122.224.48.223',
        '116.211.199.235',
        '122.14.229.102',
        '220.243.190.122',
        '122.14.229.127',
        '121.37.49.12',
        '120.46.76.152',
        '103.41.167.234',
        '223.130.200.219',
        '128.242.245.29',
        '163.53.76.86',
        '47.74.234.244',
        '172.64.151.193',
        '54.154.187.197',
        '77.88.55.60',
        '202.160.129.164',
        '52.223.41.196',
        '157.240.20.8',
        '13.226.225.2',
        '34.147.120.111',
        '151.101.66.132',
        '195.88.54.16',
        '212.77.98.9',
        '104.18.124.28',
        '46.22.183.139',
        '172.67.203.51',
        '128.242.245.253',
        '104.22.17.173',
        '104.18.13.212',
        '173.203.36.104',
        '172.67.131.165',
        '104.23.142.12',
        '103.252.115.221',
        '172.67.223.144',
        '104.18.7.65',
        '8.7.198.46',
        '54.239.28.85',
        '103.228.130.27',
        '192.0.66.108',
        '23.51.133.101',
        '31.13.83.34',
        '151.101.194.133',
        '23.227.38.33',
        '132.248.166.17',
        '3.33.172.2',
        '186.192.83.12',
        '200.147.3.157',
        '3.33.182.45',
        '172.64.151.172',
        '190.220.57.134',
        '200.12.26.11',
        '146.155.95.60',
        '45.60.131.135',
        '18.211.146.8',
        '151.101.129.91',
        '108.160.172.1',
        '202.58.60.194',
        '150.229.69.37',
        '45.60.124.46',
        '23.11.241.130',
        '18.238.192.98',
        '130.102.184.3',
        '43.245.41.7',
        '65.8.161.49',
        '203.36.148.7']


# 加载GeoLite2数据库
db_path = "D://0_master//Lab//GeoLite2-City_20240409//GeoLite2-City.mmdb"
reader = geoip2.database.Reader(db_path)

special_domain = ['rte.ie']


def get_public_ip():
    return requests.get('https://api.ipify.org').text


def get_ip_address(domain):
    return socket.gethostbyname(domain)


def get_geo_info(ip):
    response = reader.city(ip)
    return response.location.latitude, response.location.longitude


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


if __name__ == "__main__":
    # 获取本机IP地址的地理位置
    public_ip = get_public_ip()
    lat1, lon1 = get_geo_info(public_ip)
    results = []

    for domain in special_domain:
        try:
            ip_address = get_ip_address(domain)
            lat2, lon2 = get_geo_info(ip_address)
            distance = haversine(lon1, lat1, lon2, lat2)
            results.append([domain, ip_address, distance])
        except Exception as e:
            results.append([domain, 'Error', 'Error'])
            print(f"Error fetching data for {domain}: {e}")

    # 写入CSV文件

    #with open("C://Users//zpwang//Desktop//0_master//Emnets//可信泛在互联//timeChain//distances.csv", 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(['Domain', 'IP Address', 'Distance (km)'])
    #    writer.writerows(results)

    #print("Completed. Distances written to distances.csv.")