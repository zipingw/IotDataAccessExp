import subprocess
import re
import csv
import platform
import requests
from math import radians, cos, sin, asin, sqrt

def get_public_ip():
    return requests.get('https://api.ipify.org').text

def get_geo_info(ip, access_key):
    response = requests.get(f"http://api.ipstack.com/{ip}?access_key={access_key}")
    data = response.json()
    return data['latitude'], data['longitude']

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)*2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km


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
           ]  # 请根据需要填写完整列表
# 定义数据包大小列表
packet_sizes = [32, 64, 128, 256, 512, 1024]
# 定义ping次数
ping_count = 10


def ping_domain(domain, packet_size):
    """Ping域名特定次数，返回平均延迟"""
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    size_param = '-l' if platform.system().lower() == 'windows' else '-s'
    # 构建ping命令
    command = ['ping', param, '10', size_param, str(packet_size), domain]

    result = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout
    # 使用正则表达式提取平均时延
    match = re.search(r'平均 = (\d+)ms', result) or re.search(r'avg = (\d+.\d+)/', result)
    if match:
        return float(match.group(1))
    else:
        return None


def main():
    # 创建或打开CSV文件
    with open('D://0_master//Lab//ping_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV头部
        writer.writerow(['Domain', 'Packet Size (Bytes)', 'Average Delay (ms)'])

        for domain in domains:
            for size in packet_sizes:
                avg_delay = ping_domain(domain, size)
                if avg_delay:
                    print(f"Domain: {domain}, Packet size: {size} bytes, Average Delay: {avg_delay} ms")
                    # 写入数据到CSV
                    writer.writerow([domain, size, avg_delay])
                else:
                    print(f"Failed to ping {domain} with packet size {size} bytes")
                    # 对于失败的测试也写入记录，标记延迟为N/A
                    writer.writerow([domain, size, 'N/A'])


def ceju(public_ip):
    # 替换以下字符串为你的API密钥
    access_key = '<YOUR_ACCESS_KEY>'

    # 目标IP地址，此处假设为一个示例
    target_ip = '8.8.8.8'

    # 获取地理位置信息
    lat1, lon1 = get_geo_info(public_ip, access_key)
    lat2, lon2 = get_geo_info(target_ip, access_key)

    # 计算距离
    distance = haversine(lon1, lat1, lon2, lat2)
    print(f"The distance between the host and the target IP location is: {distance} kilometers.")


if __name__ == "__main__":
    # 获取运行代码的主机的公网IP
    public_ip = get_public_ip()
    ceju(public_ip)
    # main()
