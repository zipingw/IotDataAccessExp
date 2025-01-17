import numpy as np
import pandas as pd
import random
import copy
import networkx as nx
from scipy.stats import truncnorm
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import time as time_module
def cal_time(found_node, store_node):
    time4one_query = []
    refuse_cnt = 0
    accept_cnt = 0
    time_consumed_all = 0 # 最终该query消耗的总计时间
    for query in found_node:
        query_nodup = list(set(query))
        for node in query_nodup:
            flag = False
            for node_id, node_info in store_node.items():
                if flag:
                    break
                if node == node_id:
                    flag = True
                    # 首先获取当前节点的概率值， 判断是否能够命中
                    get_pro = node_info['probability']
                    if probabilistic_true(get_pro):
                        # 命中，计算时间开销
                        time_consumed = node_info['distance'] * 0.010756
                        #print(time_consumed)
                        time4one_query.append(time_consumed)
                        accept_cnt += 1
                    else:
                        # 未命中，记一次无服务次数
                        refuse_cnt += 1
        time_consumed_max = 0 # 最终该query消耗的总计时间
        for time_consumed in time4one_query:
            if time_consumed > time_consumed_max:
                time_consumed_max = time_consumed
        time4one_query.clear()
        time_consumed_all += time_consumed_max
    serve_prob = accept_cnt / (accept_cnt + refuse_cnt)
    time_consumed_all = time_consumed_all * (1 + serve_prob)
    return time_consumed_all, refuse_cnt, accept_cnt, serve_prob


def probabilistic_true(probability):
    return random.random() < probability
def init_edge_weights(G):
    count = 0
    zero_count = 0
    # 必须加上data=True才可以迭代
    for u, u_attrs in G.nodes(data=True):
        for v, v_attrs in G.nodes(data=True):
            if u != v:
                weight = 1 / (math.sqrt((u_attrs['device_id'] - v_attrs['device_id']) ** 2))
                # 添加边到图中，并附带权重信息
                G.add_edge(u, v, weight=weight)
                count += 1
    print(count)
    print(zero_count)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

# 定义截断正态分布
def truncated_normal(mean, std_dev, low, high):
    std_dev = max(std_dev, 1e-6)  # 避免标准差为零
    a = (low - mean) / std_dev
    b = (high - mean) / std_dev
    return truncnorm(a, b, loc=mean, scale=std_dev)


def generateQuery(num_queries, time_low, time_high, device_value_max):
    device_low, device_high = 1, device_value_max
    query_time_length_mu = ((1 + time_on_chain) / 6)  # 尽可能让 time_length 较长，
    query_time_length_sigma = max(int((time_on_chain - 1) / 6), 1e-6)

    queries = []

    def is_overlap_points_less_enough(time_lower_bound, time_upper_bound, device_lower_bound, device_upper_bound,
                                      queries, min_points):
        for (q_t_l, q_t_u), (q_d_l, q_d_u) in queries:
            if time_lower_bound >= q_t_u or time_upper_bound <= q_t_l or device_upper_bound <= q_d_l or device_lower_bound >= q_d_u:
                continue
            else:
                time_delta = min(abs(time_upper_bound - q_t_l), abs(time_upper_bound - time_lower_bound),
                                 abs(time_lower_bound - q_t_u), abs(q_t_l - q_t_u))
                device_delta = min(abs(device_upper_bound - q_d_l), abs(device_upper_bound - q_d_u),
                                   abs(device_upper_bound - device_lower_bound), abs(q_d_l - q_d_u))
                if (time_delta + 1) * (device_delta + 1) > min_points:
                    return False
        return True

    for _ in range(num_queries):
        bounds = []
        while (True):
            tried = 0
            mean = (time_low + time_high) / 2
            std_dev = max((time_high - time_low) / 6, 1e-6)  # 避免标准差为零, 经验法则：99.7%的数据在3个标准差内
            # 生成time_center
            time_center = round(truncated_normal(mean, std_dev, time_low, time_high).rvs())
            # print(f"time_center: {time_center}")
            time_length = abs(np.random.normal(query_time_length_mu, query_time_length_sigma))
            # print(f"time_length: {time_length}")
            time_lower_bound = round(max(time_low, time_center - time_length / 2))
            time_upper_bound = round(min(time_high, time_center + time_length / 2))

            mean = (device_low + device_high) / 2
            std_dev = max((device_high - device_low) / 6, 1e-6)  # 避免标准差为零 经验法则：99.7%的数据在3个标准差内
            # 生成time_center
            device_center = round(truncated_normal(mean, std_dev, device_low, device_high).rvs())
            # device_length = abs(np.random.normal(query_device_length_mu, query_device_length_sigma))
            device_length = round(unit_num * batch_num / time_length)
            device_lower_bound = round(max(device_low, device_center - device_length / 2))
            device_upper_bound = round(min(device_high, device_center + device_length / 2))

            area = (time_upper_bound - time_lower_bound + 1) * (device_upper_bound - device_lower_bound + 1)
            tried += 1
            # 一批数据的总数据点个数为400个, 共16个batch，得到每个batch中有25个数据点
            # 如果生成query做到对数据点全覆盖，会导致2种batch方式最终都要访问所有的batch，时间开销上差距不大，故生成16 / 2 = 8个数据点比较合适
            # 但是query也不能太少，因为query太少会导致很多节点没有出现在对图的权重进行迭代的过程中
            if 1 <= area <= 20 and time_length >= 3 and is_overlap_points_less_enough(time_lower_bound,
                                                                                      time_upper_bound,
                                                                                      device_lower_bound,
                                                                                      device_upper_bound, queries, 5):
                bounds = [(time_lower_bound, time_upper_bound), (device_lower_bound, device_upper_bound)]
                break
            if tried == 20:
                print(f"try {tried} times.")
        queries.append(bounds)
    return queries

def calLaplacianMatrix(adjacentMatrix):
    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)
    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
    # print laplacianMatrix
    # normalize
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

if __name__ == "__main__":
    rf_path = "../rt-ifttt/rt-ifttt.csv"
    rf_data = pd.read_csv(rf_path)

    weather_humidity_path = "../Historical Hourly Weather Data 2012-2017/humidity.csv"
    weather_pressure_path = "../Historical Hourly Weather Data 2012-2017/pressure.csv"
    weather_temperature_path = "../Historical Hourly Weather Data 2012-2017/temperature.csv"
    weather_wind_speed_path = "../Historical Hourly Weather Data 2012-2017/wind_speed.csv"
    weather_humidity_data = pd.read_csv(weather_humidity_path)
    weather_pressure_data = pd.read_csv(weather_pressure_path)
    weather_temperature_data = pd.read_csv(weather_temperature_path)
    weather_wind_speed_data = pd.read_csv(weather_wind_speed_path)
    weather_data = [weather_humidity_data, weather_pressure_data, weather_temperature_data, weather_wind_speed_data]


    for dataset in weather_data:
        for col_name in dataset.columns:
            # 处理空值
            if dataset[col_name].isnull().any():
                # 用平均值填充
                dataset.fillna({col_name: int(dataset[col_name].mean())}, inplace=True)
    bridge_21416_DPM_path = "../2021-04-16/2021-04-16 00-DPM.csv"
    bridge_21416_HPT_path = "../2021-04-16/2021-04-16 00-HPT.csv"
    bridge_21416_RHS_path = "../2021-04-16/2021-04-16 00-RHS.csv"
    bridge_21416_ULT_path = "../2021-04-16/2021-04-16 00-ULT.csv"
    bridge_21416_VIB_path = "../2021-04-16/2021-04-16 00-VIB.csv"
    bridge_21416_VIC_path = "../2021-04-16/2021-04-16 00-VIC.csv"
    bridge_21416_DPM_data = pd.read_csv(bridge_21416_DPM_path)
    bridge_21416_HPT_data = pd.read_csv(bridge_21416_HPT_path)
    bridge_21416_RHS_data = pd.read_csv(bridge_21416_RHS_path)
    bridge_21416_ULT_data = pd.read_csv(bridge_21416_ULT_path)
    bridge_21416_VIB_data = pd.read_csv(bridge_21416_VIB_path)
    bridge_21416_VIC_data = pd.read_csv(bridge_21416_VIC_path)

    bridge_data = [bridge_21416_DPM_data, bridge_21416_HPT_data, bridge_21416_RHS_data, bridge_21416_ULT_data,
                   bridge_21416_VIB_data, bridge_21416_VIC_data]

    for idx in range(len(weather_data)):
        weather_data[idx] = weather_data[idx].drop(weather_data[idx].columns[0], axis=1)

    weather_data_list = ["_humidity", "_pressure", "_temperature", "_wind_speed"]
    for idx_dataset in range(len(weather_data)):
        for idx in range(len(weather_data[idx_dataset].columns)):
            weather_data[idx_dataset].columns.values[idx] += weather_data_list[idx_dataset]

    data_aggregated = pd.concat([rf_data], axis=1)
    data_aggregated.columns.values[0] = "Timestamp"
    for idx in range(len(weather_data)):
        data_aggregated = pd.concat([data_aggregated, weather_data[idx].head(10000)], axis=1)

    # 模拟存储节点
    # 每个store_node 包含3个信息，distance|storage space|reputation, 并计算得到一个score
    store_node_num = 50
    store_node = {i: {'distance': 0, 'storage space': 0, 'probability': 0, 'score_sp': 0, 'score_sd': 0, 'batches': []}
                  for i in range(1, store_node_num + 1)}

    mean, std_dev = 4000, 2200  # 均值, 标准差
    gaussian_distances = np.random.normal(mean, std_dev, store_node_num)  # 生成 distance
    mean, std_dev = 1000, 160  # 均值, 标准差
    gaussian_storage_spaces = np.random.normal(mean, std_dev, store_node_num)  # 生成 storage_spaces
    mean, std_dev = 0.6, 0.1  # 均值, 标准差
    gaussian_probability = np.random.normal(mean, std_dev, store_node_num)  # 生成 storage_spaces
    gaussian_probability = np.clip(gaussian_probability, 0, 1)

    alpha_probability = 1000
    alpha_distance = 10
    # 循环遍历每个节点，为其设置高斯分布的距离
    for key, node in store_node.items():
        node['distance'] = gaussian_distances[key - 1]
        node['storage space'] = gaussian_storage_spaces[key - 1]
        node['probability'] = gaussian_probability[key - 1]
        # node['score_sp'] = node['storage space'] + alpha_probability * node['probability']
        node['score_sp'] = alpha_probability * node['probability']
        # node['score_sd'] = node['storage space'] - alpha_distance * node['distance']
        node['score_sd'] = node['distance']

    # Settings

    device_value_max = 20
    time_on_chain = 30
    time_max = 299
    # 遍历各种 unit_num与 batch_num 组合得到各组实验数据
    # 每个 unit  包含的 元素 数量
    # 每个 batch 包含的 unit 数量
    # (unit_num, batch_num)
    unit_batch_num_list = [(5, 2), (5, 4), (5, 6), (5, 8), (5, 10), (5, 12), (5, 20)]

    results = {
        '1': [],
        '10': [],
        '20': [],
        '30': [],
        '40': [],
        '50': [],
        '60': [],
        '70': [],
        '80': [],
        '90': [],
        '100': [],
    }

    unit_num, batch_num = 1, 1
    print(f"unit_num: {unit_num} batch_num: {batch_num} batch size: {unit_num * batch_num}")
    batched_basic_method = {}
    cnt = 0
    for start_time in range(0, time_max, time_on_chain):
        end_time = start_time + time_on_chain
        batch_list = []
        for time in range(start_time, end_time):
            for device in range(1, device_value_max + 1):
                # data_points.append(data_aggregated.loc[time][device])
                batch_list.append([[(time, device)]])  # 400 个数据点
        random.shuffle(batch_list)  # 随机打乱

        for batch in batch_list:
            batched_basic_method[str(cnt)] = batch
            cnt += 1

    batched_proposed_method = {}  # 以字典存储所有的Batch batch_id : []
    cnt = 0
    Query_sets = []
    for start_time in range(0, time_max, time_on_chain):
        end_time = start_time + time_on_chain
        units_list = []
        unit_tmp = []
        device_id = 1
        while device_id <= device_value_max:
            time_cnt = 0
            for time in range(start_time, end_time):
                # data_points.append(data_aggregated.loc[time][device])
                unit_tmp.append((time, device_id))
                time_cnt += 1
                if time_cnt % unit_num == 0:
                    units_list.append((device_id, copy.deepcopy(unit_tmp)))
                    unit_tmp.clear()
            device_id += 1

        num_queries = 10
        # num_queries = int(time_on_chain * device_value_max / unit_num / batch_num / 2) # 16 / 2
        # 本次生成的Query用于更新下一次的Device
        query_set = generateQuery(num_queries, start_time, end_time - 1, device_value_max)
        Query_sets.append(copy.deepcopy(query_set))  # 将每段on_chain时间内生成的query_set整合到循环外部一个总的Query集合中

        # 更新Device, 从第二批数据开始，按照query更新
        batch_list = []
        for (device_id, unit) in units_list:
            batch_list.append(copy.deepcopy(unit))

        for batch in batch_list:
            batched_proposed_method[str(cnt)] = batch
            cnt += 1

    store_node_method_r_d = copy.deepcopy(store_node)
    # batched_basic_method
    store_node_method_r_d_batched_basic_method = copy.deepcopy(store_node_method_r_d)

    for key, batch in batched_basic_method.items():
        # 要选取一个store node存储
        sorted_nodes_r_d = sorted(store_node_method_r_d_batched_basic_method.items(), key=lambda x: x[1]['score_sd'],
                                  reverse=False)
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes_r_d = sorted_nodes_r_d[:10]
        saved_flag = False
        for node in top_ten_nodes_r_d:
            if saved_flag:
                break
            get_pro = node[1]['probability']
            # 得到命中概率
            if probabilistic_true(get_pro):
                # 若命中，则存入
                store_node_method_r_d_batched_basic_method[node[0]]['batches'].append((key, batch))
                saved_flag = True
        if not saved_flag:
            # 10 个都没命中，则存入 rank 第一的 node 中
            store_node_method_r_d_batched_basic_method[top_ten_nodes_r_d[0][0]]['batches'].append((key, batch))
    # batched_proposed_method
    store_node_method_r_d_batched_proposed_method = copy.deepcopy(store_node_method_r_d)

    for key, batch in batched_proposed_method.items():
        # 要选取一个store node存储
        sorted_nodes = sorted(store_node_method_r_d_batched_proposed_method.items(), key=lambda x: x[1]['score_sd'],
                              reverse=False)  # 考虑 Distance 升序，距离小的优先
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes = sorted_nodes[:10]
        saved_flag = False
        for node in top_ten_nodes:
            if saved_flag:
                break
            get_pro = node[1]['probability']
            # 得到命中概率
            if probabilistic_true(get_pro):
                # 若命中，则存入
                store_node_method_r_d_batched_proposed_method[node[0]]['batches'].append((key, batch))  # 考虑 reputation
                saved_flag = True
        if not saved_flag:
            # 10 个都没命中，则存入 rank 第一的 node 中
            store_node_method_r_d_batched_proposed_method[top_ten_nodes[0][0]]['batches'].append((key, batch))

    # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
    found_in_basic_batch = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
    found_in_proposed_batch = []
    for query_set in Query_sets:
        basic_batches4q = []
        proposed_batches4q = []
        for query in query_set:
            # 对每一个 q, 得到 time 和 device
            time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
            # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
            # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
            # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
            # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
            batch_low, batch_high = int(time_low / time_on_chain) * int(
                time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                time_on_chain * device_value_max / unit_num / batch_num) - 1
            # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
            for device in range(device_low, device_high + 1):
                for time in range(time_low, time_high + 1):
                    # print(f"time: {time}, device: {device}")
                    basic_flag = False
                    proposed_flag = False
                    # 对每个 （time, device） 找到所属的Batch
                    for batch_id in range(batch_low, batch_high + 1):
                        # print(f"batch id is : {batch}")
                        # batch结构: [[], [], ..., []]
                        if not basic_flag:
                            batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                            for unit in batch:
                                for point in unit:
                                    if time == point[0] and device == point[1]:
                                        # print(f"----basic find point")
                                        # 找到了所属的Batch，继续寻找所属的store node
                                        basic_batches4q.append(batch_id)
                                        basic_flag = True

                        if not proposed_flag:
                            batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                            for unit in batch:

                                if time == unit[0] and device == unit[1]:
                                    # print(f"----proposed find point")
                                    # 找到了所属的Batch，继续寻找所属的store node
                                    proposed_batches4q.append(batch_id)
                                    proposed_flag = True
                        if proposed_flag and basic_flag:
                            break
                    # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
            found_in_basic_batch.append(copy.deepcopy(basic_batches4q))
            found_in_proposed_batch.append(copy.deepcopy(proposed_batches4q))
            basic_batches4q.clear()
            proposed_batches4q.clear()

    for idx in range(len(found_in_basic_batch)):
        found_in_basic_batch[idx] = list(set(found_in_basic_batch[idx]))
    for idx in range(len(found_in_proposed_batch)):
        found_in_proposed_batch[idx] = list(set(found_in_proposed_batch[idx]))

    batches_sum_basic = 0
    for item in found_in_basic_batch:
        batches_sum_basic += len(item)
    print(f"basic 的 batch 数量{batches_sum_basic}")
    batches_sum_proposed = 0
    for item in found_in_proposed_batch:
        batches_sum_proposed += len(item)
    print(f"proposed 的 batch 数量{batches_sum_proposed}")
    results[str(unit_num * batch_num)].append(0)
    results[str(unit_num * batch_num)].append((batches_sum_basic, batches_sum_proposed))
    # 现在已经找到了所有要访问的 Batch, 之后到store_node_method_r_d_batched_basic_method 和 store_node_method_r_d_batched_proposed_method 中寻找 存储了相应的 Batch 的 node
    # 要访问的每个 batch 在哪个 store node 中, 并计算累计的 1. 访问无服务次数 2. 有服务时的访问时延
    # 在Batch中 按照每5个Query为单位 存储了要访问的batch 有哪几个（会出现重复）

    # found_in_basic_batch = []      # [[], [], ..., []]
    # found_in_proposed_batch = []   # [[], [], ..., []]

    # found_in_basic_node
    found_in_basic_node_r_d = []
    basic_node_tmp = []
    for query in found_in_basic_batch:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_r_d_batched_basic_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        basic_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_basic_node_r_d.append(copy.deepcopy(basic_node_tmp))
        basic_node_tmp.clear()

    # found_in_proposed_node
    found_in_proposed_node_r_d = []
    proposed_node_tmp = []
    for query in found_in_proposed_batch:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_r_d_batched_proposed_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        proposed_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_proposed_node_r_d.append(copy.deepcopy(proposed_node_tmp))
        proposed_node_tmp.clear()
    for idx in range(len(found_in_basic_node_r_d)):
        found_in_basic_node_r_d[idx] = list(set(found_in_basic_node_r_d[idx]))
    for idx in range(len(found_in_proposed_node_r_d)):
        found_in_proposed_node_r_d[idx] = list(set(found_in_proposed_node_r_d[idx]))
    nodes_sum_basic = 0
    for item in found_in_basic_node_r_d:
        nodes_sum_basic += len(item)
    print(f"basic r_d 的 node 数量{nodes_sum_basic}")
    nodes_sum_proposed = 0
    for item in found_in_proposed_node_r_d:
        nodes_sum_proposed += len(item)
    print(f"proposed r_d 的 node 数量{nodes_sum_proposed}")
    results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))
    time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = cal_time(found_in_basic_node_r_d,
                                                                                             store_node_method_r_d_batched_basic_method)
    time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed = cal_time(
        found_in_proposed_node_r_d, store_node_method_r_d_batched_proposed_method)
    results[str(unit_num * batch_num)].append((time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
    results[str(unit_num * batch_num)].append((time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))
    print(results)
    store_node_method_r = copy.deepcopy(store_node)
    store_node_method_r_batched_basic_method = copy.deepcopy(store_node_method_r)

    for key, batch in batched_basic_method.items():
        # 要选取一个store node存储
        sorted_nodes_r = sorted(store_node_method_r_batched_basic_method.items(), key=lambda x: x[1]['score_sp'],
                                reverse=True)  # Ture 指定为降序排序
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes_r = sorted_nodes_r[:10]
        saved_flag = False
        for node in top_ten_nodes_r:
            if saved_flag:
                break
            get_pro = node[1]['probability']
            # 得到命中概率
            if probabilistic_true(get_pro):
                # 若命中，则存入
                store_node_method_r_batched_basic_method[node[0]]['batches'].append((key, batch))
                saved_flag = True
        if not saved_flag:
            # 10 个都没命中，则存入 rank 第一的 node 中
            store_node_method_r_batched_basic_method[top_ten_nodes_r[0][0]]['batches'].append((key, batch))
    store_node_method_r_batched_proposed_method = copy.deepcopy(store_node_method_r)

    for key, batch in batched_proposed_method.items():
        # 要选取一个store node存储
        sorted_nodes_r = sorted(store_node_method_r_batched_proposed_method.items(), key=lambda x: x[1]['score_sp'],
                                reverse=True)
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes_r = sorted_nodes_r[:10]
        saved_flag = False
        for node in top_ten_nodes_r:
            if saved_flag:
                break
            get_pro = node[1]['probability']
            # 得到命中概率
            if probabilistic_true(get_pro):
                # 若命中，则存入
                store_node_method_r_batched_proposed_method[node[0]]['batches'].append((key, batch))
                saved_flag = True
        if not saved_flag:
            # 10 个都没命中，则存入 rank 第一的 node 中
            store_node_method_r_batched_proposed_method[top_ten_nodes_r[0][0]]['batches'].append((key, batch))

    # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
    found_in_basic_batch_r = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
    found_in_proposed_batch_r = []
    for query_set in Query_sets:
        basic_batches4q = []
        proposed_batches4q = []
        for query in query_set:
            # 对每一个 q, 得到 time 和 device
            time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
            # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
            # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
            # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
            # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
            batch_low, batch_high = int(time_low / time_on_chain) * int(
                time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                time_on_chain * device_value_max / unit_num / batch_num) - 1
            # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
            for device in range(device_low, device_high + 1):
                for time in range(time_low, time_high + 1):
                    # print(f"time: {time}, device: {device}")
                    basic_flag = False
                    proposed_flag = False
                    # 对每个 （time, device） 找到所属的Batch
                    for batch_id in range(batch_low, batch_high + 1):
                        # print(f"batch id is : {batch}")
                        # batch结构: [[], [], ..., []]
                        if not basic_flag:
                            batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                            for unit in batch:
                                for point in unit:
                                    # print(f"{point[0]}  and  {point[1]}")
                                    if time == point[0] and device == point[1]:
                                        # print(f"----basic find point")
                                        # 找到了所属的Batch，继续寻找所属的store node
                                        basic_batches4q.append(batch_id)
                                        basic_flag = True
                        if not proposed_flag:
                            batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                            for unit in batch:
                                if time == unit[0] and device == unit[1]:
                                    # print(f"----proposed find point")
                                    # 找到了所属的Batch，继续寻找所属的store node
                                    proposed_batches4q.append(batch_id)
                                    proposed_flag = True
                        if proposed_flag and basic_flag:
                            break
                    # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
            found_in_basic_batch_r.append(copy.deepcopy(basic_batches4q))
            found_in_proposed_batch_r.append(copy.deepcopy(proposed_batches4q))
            basic_batches4q.clear()
            proposed_batches4q.clear()
    found_in_basic_node_r = []
    basic_node_tmp = []
    for query in found_in_basic_batch_r:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_r_batched_basic_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        basic_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_basic_node_r.append(copy.deepcopy(basic_node_tmp))
        basic_node_tmp.clear()

    # found_in_proposed_node
    found_in_proposed_node_r = []
    proposed_node_tmp = []
    for query in found_in_proposed_batch:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_r_batched_proposed_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        proposed_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_proposed_node_r.append(copy.deepcopy(proposed_node_tmp))
        proposed_node_tmp.clear()

    for idx in range(len(found_in_basic_node_r)):
        found_in_basic_node_r[idx] = list(set(found_in_basic_node_r[idx]))
    for idx in range(len(found_in_proposed_node_r)):
        found_in_proposed_node_r[idx] = list(set(found_in_proposed_node_r[idx]))
    nodes_sum_basic = 0
    for item in found_in_basic_node_r:
        nodes_sum_basic += len(item)
    print(f"basic r 的 node 数量{nodes_sum_basic}")
    nodes_sum_proposed = 0
    for item in found_in_proposed_node_r:
        nodes_sum_proposed += len(item)
    print(f"proposed r 的 node 数量{nodes_sum_proposed}")
    results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))

    time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = cal_time(found_in_basic_node_r,
                                                                                             store_node_method_r_batched_basic_method)
    time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed = cal_time(
        found_in_proposed_node_r, store_node_method_r_batched_proposed_method)
    results[str(unit_num * batch_num)].append((time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
    results[str(unit_num * batch_num)].append((time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))

    store_node_method_d = copy.deepcopy(store_node)
    # batched_basic_method
    store_node_method_d_batched_basic_method = copy.deepcopy(store_node_method_d)

    for key, batch in batched_basic_method.items():
        # 要选取一个store node存储
        sorted_nodes_d = sorted(store_node_method_d_batched_basic_method.items(), key=lambda x: x[1]['score_sd'],
                                reverse=False)  # 升序排序，距离越小越优先
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes = sorted_nodes_d[:10]
        random_choice_node = random.choice(top_ten_nodes)
        store_node_method_d_batched_basic_method[random_choice_node[0]]['batches'].append((key, batch))

    store_node_method_d_batched_proposed_method = copy.deepcopy(store_node_method_d)

    for key, batch in batched_proposed_method.items():
        # 要选取一个store node存储
        sorted_nodes_d = sorted(store_node_method_d_batched_proposed_method.items(), key=lambda x: x[1]['score_sd'],
                                reverse=False)  # 升序排序，距离越小越优先
        # 从得分前 10 的节点中随机选择一个节点
        top_ten_nodes = sorted_nodes_d[:10]
        random_choice_node = random.choice(top_ten_nodes)
        store_node_method_d_batched_proposed_method[random_choice_node[0]]['batches'].append((key, batch))

    # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
    found_in_basic_batch = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
    found_in_proposed_batch = []
    found_in_basic_batch.clear()
    found_in_proposed_batch.clear()
    for query_set in Query_sets:
        basic_batches4q = []
        proposed_batches4q = []
        for query in query_set:
            # 对每一个 q, 得到 time 和 device
            time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
            # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
            # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
            # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
            # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
            batch_low, batch_high = int(time_low / time_on_chain) * int(
                time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                time_on_chain * device_value_max / unit_num / batch_num) - 1
            # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
            for device in range(device_low, device_high + 1):
                for time in range(time_low, time_high + 1):
                    # print(f"time: {time}, device: {device}")
                    basic_flag = False
                    proposed_flag = False
                    # 对每个 （time, device） 找到所属的Batch
                    for batch_id in range(batch_low, batch_high + 1):
                        # print(f"batch id is : {batch}")
                        # batch结构: [[], [], ..., []]

                        if not basic_flag:
                            batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                            for unit in batch:
                                for point in unit:
                                    # print(f"{point[0]}  and  {point[1]}")
                                    if time == point[0] and device == point[1]:
                                        # print(f"----basic find point")
                                        # 找到了所属的Batch，继续寻找所属的store node
                                        basic_batches4q.append(batch_id)
                                        basic_flag = True

                        if not proposed_flag:
                            batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                            for unit in batch:
                                if time == unit[0] and device == unit[1]:
                                    # print(f"----proposed find point")
                                    # 找到了所属的Batch，继续寻找所属的store node
                                    proposed_batches4q.append(batch_id)
                                    proposed_flag = True
                        if proposed_flag and basic_flag:
                            break
                    # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
            found_in_basic_batch.append(copy.deepcopy(basic_batches4q))
            found_in_proposed_batch.append(copy.deepcopy(proposed_batches4q))
            basic_batches4q.clear()
            proposed_batches4q.clear()
    found_in_basic_node_d = []
    basic_node_tmp = []
    for query in found_in_basic_batch:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_d_batched_basic_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        basic_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_basic_node_d.append(copy.deepcopy(basic_node_tmp))
        basic_node_tmp.clear()

    # found_in_proposed_node
    found_in_proposed_node_d = []
    proposed_node_tmp = []
    for query in found_in_proposed_batch:
        # print(query)
        found_batches = list(set(query))
        # print(found_batches)
        for batch_id in found_batches:
            # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
            flag = False
            for node_id, node_info in store_node_method_d_batched_proposed_method.items():
                if flag:
                    break
                for key, value in node_info['batches']:
                    if int(batch_id) == int(key):
                        proposed_node_tmp.append(node_id)
                        flag = True
                        break
        found_in_proposed_node_d.append(copy.deepcopy(proposed_node_tmp))
        proposed_node_tmp.clear()

    for idx in range(len(found_in_basic_node_d)):
        found_in_basic_node_d[idx] = list(set(found_in_basic_node_d[idx]))
    for idx in range(len(found_in_proposed_node_d)):
        found_in_proposed_node_d[idx] = list(set(found_in_proposed_node_d[idx]))

    nodes_sum_basic = 0
    for item in found_in_basic_node_r:
        nodes_sum_basic += len(item)
    print(f"basic r 的 node 数量{nodes_sum_basic}")
    nodes_sum_proposed = 0
    for item in found_in_proposed_node_r:
        nodes_sum_proposed += len(item)
    print(f"proposed r 的 node 数量{nodes_sum_proposed}")
    results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))
    time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = cal_time(found_in_basic_node_d,
                                                                                             store_node_method_d_batched_basic_method)
    time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed = cal_time(
        found_in_proposed_node_d, store_node_method_d_batched_proposed_method)
    results[str(unit_num * batch_num)].append((time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
    results[str(unit_num * batch_num)].append((time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))

    print(results)

    result_temp = {}
    for key, value in results.items():
        if len(results[key]) != 0:
            result_temp[key] = value
    # 将数据转换为DataFrame
    df = pd.DataFrame(result_temp)
    # 将DataFrame写入到Excel文件
    output_file = f'./output_batch_size_{unit_num * batch_num}.xlsx'
    df.to_excel(output_file, index=False)
    print(f'Data has been written to {output_file}')

    for (unit_num, batch_num) in unit_batch_num_list:
        print(f"unit_num: {unit_num} batch_num: {batch_num} batch size: {unit_num * batch_num}")
        batched_basic_method = {}  # 以字典存储所有的Batch batch_id : []
        cnt = 0
        for start_time in range(0, time_max, time_on_chain):
            end_time = start_time + time_on_chain

            data_points = []
            for time in range(start_time, end_time):
                for device in range(1, device_value_max + 1):
                    # data_points.append(data_aggregated.loc[time][device])
                    data_points.append((time, device))  # 400 个数据点
            random.shuffle(data_points)  # 随机打乱

            # 按照 unit_num 个 data_point 为一组打包
            units_list = []
            unit_tmp = []
            # print(len(data_points))
            for idx in range(0, len(data_points), unit_num):  # 以 5 为步长
                for i in range(idx, idx + unit_num):
                    unit_tmp.append(data_points[i])
                units_list.append(copy.deepcopy(unit_tmp))  # 得到 400 / 5 = 80 个 unit
                unit_tmp.clear()
            # print(len(units_list))
            random.shuffle(units_list)

            # 按照 batch_num 个 unit 为一组打包
            batch_list = []
            batch_tmp = []
            for idx in range(0, len(units_list), batch_num):  # 得到 80 / 5 = 16 个 batch
                for i in range(idx, idx + batch_num):
                    batch_tmp.append(units_list[i])
                batch_list.append(copy.deepcopy(batch_tmp))
                batch_tmp.clear()
            for batch in batch_list:
                batched_basic_method[str(cnt)] = batch
                cnt += 1

        G = nx.Graph()
        G.add_nodes_from(range(device_value_max))

        vDevice = 1
        count = 0
        for node in G.nodes():
            # 为每个节点添加两个属性值
            nx.set_node_attributes(G, {node: {'device_id': vDevice}})
            count += 1
            if vDevice <= device_value_max:
                vDevice += 1

        init_edge_weights(G)
        batched_proposed_method = {}  # 以字典存储所有的Batch batch_id : []
        cnt = 0

        # 采样记录所有的Query
        Query_sets = []
        exec_time_list = []
        for start_time in range(0, time_max, time_on_chain):
            end_time = start_time + time_on_chain
            units_list = []
            unit_tmp = []
            device_id = 1
            while device_id <= device_value_max:
                time_cnt = 0
                for time in range(start_time, end_time):
                    # data_points.append(data_aggregated.loc[time][device])
                    unit_tmp.append((time, device_id))
                    time_cnt += 1
                    if time_cnt % unit_num == 0:
                        units_list.append((device_id, copy.deepcopy(unit_tmp)))
                        unit_tmp.clear()
                device_id += 1

            # 需要生成Query 并更新Device图 再进行分类，最后根据谱聚类结果 按照10个unit为一组打包为Batch
            alpha = 0.8
            beta = 0.5  # beta 比较适合 * 一个信息表示该此访问的 强度

            num_queries = 10
            # num_queries = int(time_on_chain * device_value_max / unit_num / batch_num / 2) # 16 / 2
            # 本次生成的Query用于更新下一次的Device
            query_set = generateQuery(num_queries, start_time, end_time - 1, device_value_max)
            Query_sets.append(copy.deepcopy(query_set))  # 将每段on_chain时间内生成的query_set整合到循环外部一个总的Query集合中

            # 更新Device, 从第二批数据开始，按照query更新

            if start_time != 0:
                # 记录开始时间
                ratioCut_start_time = time_module.time()
                for query in Query_sets[-1]:
                    queried_nodes = [node for node, attributes in G.nodes(data=True) if attributes['device_id'] >= query[1][0] and attributes['device_id'] <= query[1][1]]
                    for u, v, attrs in G.edges(data=True):
                        if u in queried_nodes and v in queried_nodes:
                            attrs['weight'] = alpha * G.edges[u, v]['weight'] + beta * G.edges[v, u]['weight']
                        else:
                            attrs['weight'] = alpha * G.edges[u, v]['weight']

            print(f"{unit_num * batch_num} : 迭代图完成")
            # 谱聚类
            Adjacent = nx.adjacency_matrix(G)  # 获取邻接矩阵
            total_sum = np.sum(Adjacent.data)  # 对稀疏矩阵中的每个非零元素进行标准化
            normalized_adjacent = Adjacent / total_sum  # normalized_adjacent = normalized_adjacent * 1e6
            # print("包含 NaN:", np.isnan(normalized_adjacent).any())
            Laplacian = calLaplacianMatrix(normalized_adjacent)
            Laplacian = Laplacian.astype(np.float64)
            # print("包含 NaN:", np.isnan(Laplacian).any())
            Laplacian = np.nan_to_num(Laplacian, nan=0.0)

            x, V = np.linalg.eig(Laplacian)
            x = zip(x, range(len(x)))
            x = sorted(x, key=lambda x: x[0])
            H = np.vstack([V[:, i] for (v, i) in x[:device_value_max]]).T
            class_number = 10
            # class_number = int(time_on_chain * device_value_max / unit_num / batch_num) # 类别数量 4
            sp_kmeans = KMeans(n_clusters=class_number, n_init='auto').fit(H)
            # sp_kmeans = KMeans(n_clusters=16, n_init='auto').fit(H)
            labels = sp_kmeans.labels_  # labels 标记了每一个 device 所属的类别
            nlist = list(G)  # 20 个 device
            print(f"{unit_num * batch_num} : 谱聚类完成")
            if start_time != 0:
                # 记录结束时间
                ratioCut_end_time = time_module.time()
                # 计算运行时间
                execution_time = ratioCut_end_time - ratioCut_start_time
                execution_time_ms = execution_time * 1000
                exec_time_list.append(execution_time_ms)

            node2label = {}
            for idx in range(len(nlist)):
                node2label[str(nlist[idx] + 1)] = labels[idx]

            # 统计列表中每个类别的数量
            number_counts = Counter(labels)
            class_counts = []

            # 由于现在只要对unit 组合为 Batch ，而一个 unit 内的 data_point 都属于同一个 Device ，故只需要找到 20个 Device 与 类别 的映射关系
            units_with_label = []
            for (device_id, unit) in units_list:
                label = node2label[str(device_id)]
                units_with_label.append((label, copy.deepcopy(unit)))

            # 将unit组合为Batch
            batch_list = []
            batch_tmp = []
            # 20个 Device 被归为了4类，现在遍历所有的unit , 根据unit中的data point的 device_id 决定其所属的类别 class
            batch_cnt = 0
            units_with_label.sort(key=lambda x: x[0])  # 按照labels对units进行升序排序
            for (label, unit) in units_with_label:
                batch_tmp.append(copy.deepcopy(unit))
                batch_cnt += 1
                if batch_cnt % batch_num == 0:
                    batch_list.append(copy.deepcopy(batch_tmp))
                    batch_tmp.clear()

            for batch in batch_list:
                batched_proposed_method[str(cnt)] = batch
                cnt += 1
            print(f"{unit_num * batch_num} : batched_proposed_method完成")
        # 计算出RatioCut所需的平均时间
        sum = 0.0
        for item in exec_time_list:
            sum += item
        exec_time_avg = sum / len(exec_time_list)
        print(exec_time_avg)
        results[str(unit_num * batch_num)].append(exec_time_avg)

        # 方法一:

        store_node_method_r_d = copy.deepcopy(store_node)
        # batched_basic_method
        store_node_method_r_d_batched_basic_method = copy.deepcopy(store_node_method_r_d)

        for key, batch in batched_basic_method.items():
            # 要选取一个store node存储
            sorted_nodes_r_d = sorted(store_node_method_r_d_batched_basic_method.items(),
                                      key=lambda x: x[1]['score_sd'], reverse=False)
            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes_r_d = sorted_nodes_r_d[:10]

            saved_flag = False
            for node in top_ten_nodes_r_d:
                if saved_flag:
                    break
                get_pro = node[1]['probability']
                # 得到命中概率
                if probabilistic_true(get_pro):
                    # 若命中，则存入
                    store_node_method_r_d_batched_basic_method[node[0]]['batches'].append((key, batch))
                    saved_flag = True
            if not saved_flag:
                # 10 个都没命中，则存入 rank 第一的 node 中
                store_node_method_r_d_batched_basic_method[top_ten_nodes_r_d[0][0]]['batches'].append((key, batch))

        # batched_proposed_method
        store_node_method_r_d_batched_proposed_method = copy.deepcopy(store_node_method_r_d)

        for key, batch in batched_proposed_method.items():
            # 要选取一个store node存储
            sorted_nodes = sorted(store_node_method_r_d_batched_proposed_method.items(), key=lambda x: x[1]['score_sd'],
                                  reverse=False)  # 考虑 Distance 升序，距离小的优先
            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes = sorted_nodes[:10]
            saved_flag = False
            for node in top_ten_nodes:
                if saved_flag:
                    break
                get_pro = node[1]['probability']
                # 得到命中概率
                if probabilistic_true(get_pro):
                    # 若命中，则存入
                    store_node_method_r_d_batched_proposed_method[node[0]]['batches'].append(
                        (key, batch))  # 考虑 reputation
                    saved_flag = True
            if not saved_flag:
                # 10 个都没命中，则存入 rank 第一的 node 中
                store_node_method_r_d_batched_proposed_method[top_ten_nodes[0][0]]['batches'].append((key, batch))

        # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
        found_in_basic_batch = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
        found_in_proposed_batch = []
        for query_set in Query_sets:
            basic_batches4q = []
            proposed_batches4q = []
            for query in query_set:
                # 对每一个 q, 得到 time 和 device
                time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
                # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
                # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
                # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
                # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
                batch_low, batch_high = int(time_low / time_on_chain) * int(
                    time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                    time_on_chain * device_value_max / unit_num / batch_num) - 1
                # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
                for device in range(device_low, device_high + 1):
                    for time in range(time_low, time_high + 1):
                        # print(f"time: {time}, device: {device}")
                        basic_flag = False
                        proposed_flag = False
                        # 对每个 （time, device） 找到所属的Batch
                        for batch_id in range(batch_low, batch_high + 1):
                            # print(f"batch id is : {batch}")
                            # batch结构: [[], [], ..., []]

                            if not basic_flag:
                                batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        # print(f"{point[0]}  and  {point[1]}")
                                        if time == point[0] and device == point[1]:
                                            # print(f"----basic find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            basic_batches4q.append(batch_id)
                                            basic_flag = True

                            if not proposed_flag:
                                batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        if time == point[0] and device == point[1]:
                                            # print(f"----proposed find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            proposed_batches4q.append(batch_id)
                                            proposed_flag = True
                            if proposed_flag and basic_flag:
                                break
                        # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
                found_in_basic_batch.append(copy.deepcopy(basic_batches4q))
                found_in_proposed_batch.append(copy.deepcopy(proposed_batches4q))
                basic_batches4q.clear()
                proposed_batches4q.clear()

        for idx in range(len(found_in_basic_batch)):
            found_in_basic_batch[idx] = list(set(found_in_basic_batch[idx]))

        for idx in range(len(found_in_proposed_batch)):
            found_in_proposed_batch[idx] = list(set(found_in_proposed_batch[idx]))

        batches_sum_basic = 0
        for item in found_in_basic_batch:
            batches_sum_basic += len(item)
        print(f"basic 的 batch 数量{batches_sum_basic}")

        batches_sum_proposed = 0
        for item in found_in_proposed_batch:
            batches_sum_proposed += len(item)
        print(f"proposed 的 batch 数量{batches_sum_proposed}")

        results[str(unit_num * batch_num)].append((batches_sum_basic, batches_sum_proposed))

        # found_in_basic_node
        found_in_basic_node_r_d = []
        basic_node_tmp = []
        for query in found_in_basic_batch:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_r_d_batched_basic_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            basic_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_basic_node_r_d.append(copy.deepcopy(basic_node_tmp))
            basic_node_tmp.clear()

        # found_in_proposed_node
        found_in_proposed_node_r_d = []
        proposed_node_tmp = []
        for query in found_in_proposed_batch:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_r_d_batched_proposed_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            proposed_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_proposed_node_r_d.append(copy.deepcopy(proposed_node_tmp))
            proposed_node_tmp.clear()

        for idx in range(len(found_in_basic_node_r_d)):
            found_in_basic_node_r_d[idx] = list(set(found_in_basic_node_r_d[idx]))
        for idx in range(len(found_in_proposed_node_r_d)):
            found_in_proposed_node_r_d[idx] = list(set(found_in_proposed_node_r_d[idx]))

        nodes_sum_basic = 0
        for item in found_in_basic_node_r_d:
            nodes_sum_basic += len(item)
        print(f"basic r_d 的 node 数量{nodes_sum_basic}")
        nodes_sum_proposed = 0
        for item in found_in_proposed_node_r_d:
            nodes_sum_proposed += len(item)
        print(f"proposed r_d 的 node 数量{nodes_sum_proposed}")
        results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))

        time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = cal_time(
            found_in_basic_node_r_d, store_node_method_r_d_batched_basic_method)
        time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed = cal_time(
            found_in_proposed_node_r_d, store_node_method_r_d_batched_proposed_method)
        results[str(unit_num * batch_num)].append((time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
        results[str(unit_num * batch_num)].append((time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))

        # 方法二:
        store_node_method_r = copy.deepcopy(store_node)
        store_node_method_r_batched_basic_method = copy.deepcopy(store_node_method_r)

        for key, batch in batched_basic_method.items():
            # 要选取一个store node存储
            sorted_nodes_r = sorted(store_node_method_r_batched_basic_method.items(), key=lambda x: x[1]['score_sp'],
                                    reverse=True)  # Ture 指定为降序排序

            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes_r = sorted_nodes_r[:10]
            saved_flag = False
            for node in top_ten_nodes_r:
                if saved_flag:
                    break
                get_pro = node[1]['probability']
                # 得到命中概率
                if probabilistic_true(get_pro):
                    # 若命中，则存入
                    store_node_method_r_batched_basic_method[node[0]]['batches'].append((key, batch))
                    saved_flag = True

            if not saved_flag:
                # 10 个都没命中，则存入 rank 第一的 node 中
                store_node_method_r_batched_basic_method[top_ten_nodes_r[0][0]]['batches'].append((key, batch))

        store_node_method_r_batched_proposed_method = copy.deepcopy(store_node_method_r)

        for key, batch in batched_proposed_method.items():
            # 要选取一个store node存储
            sorted_nodes_r = sorted(store_node_method_r_batched_proposed_method.items(), key=lambda x: x[1]['score_sp'],
                                    reverse=True)
            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes_r = sorted_nodes_r[:10]
            saved_flag = False
            for node in top_ten_nodes_r:
                if saved_flag:
                    break
                get_pro = node[1]['probability']
                # 得到命中概率
                if probabilistic_true(get_pro):
                    # 若命中，则存入
                    store_node_method_r_batched_proposed_method[node[0]]['batches'].append((key, batch))
                    saved_flag = True
            if not saved_flag:
                # 10 个都没命中，则存入 rank 第一的 node 中
                store_node_method_r_batched_proposed_method[top_ten_nodes_r[0][0]]['batches'].append((key, batch))

        # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
        found_in_basic_batch_r = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
        found_in_proposed_batch_r = []
        for query_set in Query_sets:
            basic_batches4q = []
            proposed_batches4q = []
            for query in query_set:
                # 对每一个 q, 得到 time 和 device
                time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
                # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
                # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
                # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
                # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
                batch_low, batch_high = int(time_low / time_on_chain) * int(
                    time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                    time_on_chain * device_value_max / unit_num / batch_num) - 1
                # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
                for device in range(device_low, device_high + 1):
                    for time in range(time_low, time_high + 1):
                        # print(f"time: {time}, device: {device}")
                        basic_flag = False
                        proposed_flag = False
                        # 对每个 （time, device） 找到所属的Batch
                        for batch_id in range(batch_low, batch_high + 1):
                            # print(f"batch id is : {batch}")
                            # batch结构: [[], [], ..., []]
                            if not basic_flag:
                                batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        # print(f"{point[0]}  and  {point[1]}")
                                        if time == point[0] and device == point[1]:
                                            # print(f"----basic find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            basic_batches4q.append(batch_id)
                                            basic_flag = True
                            if not proposed_flag:
                                batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        if time == point[0] and device == point[1]:
                                            # print(f"----proposed find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            proposed_batches4q.append(batch_id)
                                            proposed_flag = True
                            if proposed_flag and basic_flag:
                                break
                        # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
                found_in_basic_batch_r.append(copy.deepcopy(basic_batches4q))
                found_in_proposed_batch_r.append(copy.deepcopy(proposed_batches4q))
                basic_batches4q.clear()
                proposed_batches4q.clear()

        for idx in range(len(found_in_basic_batch_r)):
            found_in_basic_batch_r[idx] = list(set(found_in_basic_batch_r[idx]))

        for idx in range(len(found_in_proposed_batch_r)):
            found_in_proposed_batch_r[idx] = list(set(found_in_proposed_batch_r[idx]))

        found_in_basic_node_r = []
        basic_node_tmp = []
        for query in found_in_basic_batch_r:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_r_batched_basic_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            basic_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_basic_node_r.append(copy.deepcopy(basic_node_tmp))
            basic_node_tmp.clear()

        # found_in_proposed_node
        found_in_proposed_node_r = []
        proposed_node_tmp = []
        for query in found_in_proposed_batch:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_r_batched_proposed_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            proposed_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_proposed_node_r.append(copy.deepcopy(proposed_node_tmp))
            proposed_node_tmp.clear()

        for idx in range(len(found_in_basic_node_r)):
            found_in_basic_node_r[idx] = list(set(found_in_basic_node_r[idx]))

        for idx in range(len(found_in_proposed_node_r)):
            found_in_proposed_node_r[idx] = list(set(found_in_proposed_node_r[idx]))

        nodes_sum_basic = 0
        for item in found_in_basic_node_r:
            nodes_sum_basic += len(item)
        print(f"basic r 的 node 数量{nodes_sum_basic}")
        nodes_sum_proposed = 0
        for item in found_in_proposed_node_r:
            nodes_sum_proposed += len(item)
        print(f"proposed r 的 node 数量{nodes_sum_proposed}")

        results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))

        time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = cal_time(found_in_basic_node_r, store_node_method_r_batched_basic_method)
        time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed = cal_time(found_in_proposed_node_r, store_node_method_r_batched_proposed_method)

        results[str(unit_num * batch_num)].append((time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
        results[str(unit_num * batch_num)].append((time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))


        # 方法三
        store_node_method_d = copy.deepcopy(store_node)

        # batched_basic_method
        store_node_method_d_batched_basic_method = copy.deepcopy(store_node_method_d)

        for key, batch in batched_basic_method.items():
            # 要选取一个store node存储
            sorted_nodes_d = sorted(store_node_method_d_batched_basic_method.items(), key=lambda x: x[1]['score_sd'],
                                    reverse=False)  # 升序排序，距离越小越优先
            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes = sorted_nodes_d[:10]
            random_choice_node = random.choice(top_ten_nodes)
            store_node_method_d_batched_basic_method[random_choice_node[0]]['batches'].append((key, batch))

        store_node_method_d_batched_proposed_method = copy.deepcopy(store_node_method_d)

        for key, batch in batched_proposed_method.items():
            # 要选取一个store node存储
            sorted_nodes_d = sorted(store_node_method_d_batched_proposed_method.items(), key=lambda x: x[1]['score_sd'],
                                    reverse=False)  # 升序排序，距离越小越优先
            # 从得分前 10 的节点中随机选择一个节点
            top_ten_nodes = sorted_nodes_d[:10]
            random_choice_node = random.choice(top_ten_nodes)
            store_node_method_d_batched_proposed_method[random_choice_node[0]]['batches'].append((key, batch))

        # 遍历Query_sets，在两种store node的存储方式上，访问所有的query，计算得到指标
        found_in_basic_batch = []  # [[], [], ..., []] 记录了每个 query 中每个 point 所在的batch_id ， 用于后续去 store
        found_in_proposed_batch = []
        found_in_basic_batch.clear()
        found_in_proposed_batch.clear()
        for query_set in Query_sets:
            basic_batches4q = []
            proposed_batches4q = []
            for query in query_set:
                # 对每一个 q, 得到 time 和 device
                time_low, time_high, device_low, device_high = query[0][0], query[0][1], query[1][0], query[1][1]
                # print(f"time_low:{time_low}, time_high:{time_high}, device_low:{device_low}, device_high:{device_high}")
                # 找到 device 所在的 batch, 计算出每个 q 需要访问哪些 batch
                # 直接遍历搜索空间过大，进行优化: 从time来确定Batch_id范围: 0~19s 内的数据处于 0~3 的 batch 之间
                # 由于 同一批Query_set中的元素都在同一批Batch中，故直接由time_low定出 batch range
                batch_low, batch_high = int(time_low / time_on_chain) * int(
                    time_on_chain * device_value_max / unit_num / batch_num), (int(time_low / time_on_chain) + 1) * int(
                    time_on_chain * device_value_max / unit_num / batch_num) - 1
                # print(f"batch_low:{batch_low} and batch_high:{batch_high}")
                for device in range(device_low, device_high + 1):
                    for time in range(time_low, time_high + 1):
                        # print(f"time: {time}, device: {device}")
                        basic_flag = False
                        proposed_flag = False
                        # 对每个 （time, device） 找到所属的Batch
                        for batch_id in range(batch_low, batch_high + 1):
                            # print(f"batch id is : {batch}")
                            # batch结构: [[], [], ..., []]

                            if not basic_flag:
                                batch = copy.deepcopy(batched_basic_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        # print(f"{point[0]}  and  {point[1]}")
                                        if time == point[0] and device == point[1]:
                                            # print(f"----basic find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            basic_batches4q.append(batch_id)
                                            basic_flag = True

                            if not proposed_flag:
                                batch = copy.deepcopy(batched_proposed_method[str(batch_id)])
                                for unit in batch:
                                    for point in unit:
                                        if time == point[0] and device == point[1]:
                                            # print(f"----proposed find point")
                                            # 找到了所属的Batch，继续寻找所属的store node
                                            proposed_batches4q.append(batch_id)
                                            proposed_flag = True
                            if proposed_flag and basic_flag:
                                break
                        # print(f"belong_to_basic_batch:{belong_to_basic_batch} and belong_to_proposed_batch:{belong_to_proposed_batch}")
                found_in_basic_batch.append(copy.deepcopy(basic_batches4q))
                found_in_proposed_batch.append(copy.deepcopy(proposed_batches4q))
                basic_batches4q.clear()
                proposed_batches4q.clear()

        for idx in range(len(found_in_basic_batch)):
            found_in_basic_batch[idx] = list(set(found_in_basic_batch[idx]))
        for idx in range(len(found_in_proposed_batch)):
            found_in_proposed_batch[idx] = list(set(found_in_proposed_batch[idx]))

        found_in_basic_node_d = []
        basic_node_tmp = []
        for query in found_in_basic_batch:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_d_batched_basic_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            basic_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_basic_node_d.append(copy.deepcopy(basic_node_tmp))
            basic_node_tmp.clear()

        # found_in_proposed_node
        found_in_proposed_node_d = []
        proposed_node_tmp = []
        for query in found_in_proposed_batch:
            # print(query)
            found_batches = list(set(query))
            # print(found_batches)
            for batch_id in found_batches:
                # 首先要合并所有相同的 batch_id , 再对每个不同的Batch_id到store_node中找到 node_id 存入found_in_xxx_node
                flag = False
                for node_id, node_info in store_node_method_d_batched_proposed_method.items():
                    if flag:
                        break
                    for key, value in node_info['batches']:
                        if int(batch_id) == int(key):
                            proposed_node_tmp.append(node_id)
                            flag = True
                            break
            found_in_proposed_node_d.append(copy.deepcopy(proposed_node_tmp))
            proposed_node_tmp.clear()

        for idx in range(len(found_in_basic_node_d)):
            found_in_basic_node_d[idx] = list(set(found_in_basic_node_d[idx]))
        for idx in range(len(found_in_proposed_node_d)):
            found_in_proposed_node_d[idx] = list(set(found_in_proposed_node_d[idx]))

        nodes_sum_basic = 0
        for item in found_in_basic_node_d:
            nodes_sum_basic += len(item)
        print(f"basic r 的 node 数量{nodes_sum_basic}")
        nodes_sum_proposed = 0
        for item in found_in_proposed_node_d:
            nodes_sum_proposed += len(item)
        print(f"proposed r 的 node 数量{nodes_sum_proposed}")

        results[str(unit_num * batch_num)].append((nodes_sum_basic, nodes_sum_proposed))

        time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic = (
            cal_time(found_in_basic_node_d, store_node_method_d_batched_basic_method))
        time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed =\
            cal_time(found_in_proposed_node_d, store_node_method_d_batched_proposed_method)

        results[str(unit_num * batch_num)].append(
            (time_consumed_max_basic, refuse_cnt_basic, accept_cnt_basic, serve_prob_basic))
        results[str(unit_num * batch_num)].append(
            (time_consumed_max_proposed, refuse_cnt_proposed, accept_cnt_proposed, serve_prob_proposed))

        result_temp = {}
        for key, value in results.items():
            if len(results[key]) != 0:
                result_temp[key] = value
        df = pd.DataFrame(result_temp)
        # 将DataFrame写入到Excel文件
        output_file = f'./output_batch_size_{unit_num * batch_num}.xlsx'
        df.to_excel(output_file, index=False)
        print(f'Data has been written to {output_file}')
