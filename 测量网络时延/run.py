import subprocess
import platform

# 定义目标IP地址列表
target_ips = []
# ips ["baidu.com", "weibo.com", "jd.com", "163.com", "alibabagroup.com", "weixin.qq.com",
# "v.qq.com", "iqiyi.com", "bytedance.com", "feishu.cn", "douyin.com", "huawei.com", "csdn.net", "juejin.cn",
# "zhihu.com"]
# ["39.156.66.10", , "111.13.149.108", "123.58.180.7", "47.246.136.170", "223.109.215.29",
# "112.13.84.40", "111.206.13.64", "122.14.229.7", "220.243.190.122", "122.14.229.128", "121.37.49.12",
# "120.46.76.152", "117.149.155.244", "103.41.167.234"]
# target_hops = [18, 14, 20, 10, 21, 13,
#               13, 14, 18, 16, 18, 18,
#               18, 11, 22, ]

# ips ["openai.com", "youtube.com", "twitter.com",
# "bitcoin.org", "ethereum.org", "solana.com", "protocol.ai",
# "binance.com", "aptos.com", "nextjs.org"]
# 定义数据包大小列表，单位为字节
packet_sizes = [32]


def run_ping(ip, packet_size):
    """运行ping命令"""
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    size_param = '-l' if platform.system().lower() == 'windows' else '-s'
    command = ['ping', param, '10', size_param, str(packet_size), ip]
    return subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout


def run_traceroute(ip):
    """运行traceroute命令"""
    command = ['traceroute', ip] if platform.system().lower() != 'windows' else ['tracert', ip]
    return subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout


def run():
    for ip in target_ips:
        # print(f"Testing IP: {ip}\n")
        # 获取路由跳数
        print("Traceroute result:")
        # traceroute_result = run_traceroute(ip)
        # print(traceroute_result)
        # print(traceroute_result[5:7])
        # target_hops.append(traceroute_result[5:7])
        print("\n")
    # print(target_hops)


'''        # 测试不同大小的数据包
        for size in packet_sizes:
            print(f"Ping with packet size {size} bytes")
            ping_result = run_ping(ip, size)
            print(ping_result)'''


if __name__ == "__main__":
    run()
