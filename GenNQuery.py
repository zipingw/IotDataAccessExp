import numpy as np
import time
import pandas as pd
from datetime import datetime


def convert_datetime_to_timestamp(file_path, output_file_path):
    try:
        # Load the data from CSV
        data = pd.read_csv(file_path)

        # Convert the first column 'datetime' to datetime and then to Unix timestamp
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime']).astype('int64') // 10 ** 9
        else:
            # If the first column has a different name, replace 'datetime' with the actual column name
            data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0]).astype('int64') // 10 ** 9

        # Save the modified data to a new CSV file
        data.to_csv(output_file_path, index=False)
        return f"Data successfully converted and saved to {output_file_path}"
    except Exception as e:
        return f"An error occurred: {e}"

# 载入数据集 替代生成数据集
# 1. 根据数据集信息得到共有几个维度，共有多少条数据，每一维度的数据范围上下限是多少
# 2. 修改生成 Query 的均值和方差得到一个合适的值 用于查询


def generate_data_set(num_points=10000, n_dim=10):
    # 生成数据集，每个维度有不同的范围
    data_ranges = [(np.random.uniform(0, 50), np.random.uniform(50, 100)) for _ in range(n_dim)]
    data_set = np.array([np.random.uniform(low, high, num_points) for low, high in data_ranges]).T
    return data_set, data_ranges


def generate_n_dim_queries(num_queries=10, data_ranges=None):
    query_length_mu = 20  # 增加查询长度的平均值
    query_length_sigma = 5  # 减少查询长度的标准差

    queries = []
    for _ in range(num_queries):
        bounds = []
        for low, high in data_ranges:
            mu = (high + low) / 2
            sigma = (high - low) / 12  # 使用 12.5% 的范围作为标准差
            center = np.random.normal(mu, sigma)
            length = abs(np.random.normal(query_length_mu, query_length_sigma))
            lower_bound = max(low, center - length / 2)
            upper_bound = min(high, center + length / 2)
            bounds.append((lower_bound, upper_bound))
        queries.append(bounds)
    return queries


def query_data_set(data_set, queries):
    results = []
    for query in queries:
        condition = np.ones(len(data_set), dtype=bool)
        for dim, (lower_bound, upper_bound) in enumerate(query):
            condition &= (data_set[:, dim] >= lower_bound) & (data_set[:, dim] <= upper_bound)
        results.append(data_set[condition])
    return results


if __name__ == "__main__":
    # change datetime to int
    '''input_file_path = 'D://0_master//Lab//datasets//Historical Hourly Weather Data 2012-2017//humidity.csv'
    output_file_path = 'D://0_master//Lab//datasets//Historical Hourly Weather Data 2012-2017//humidity_changed.csv'
    result_message = convert_datetime_to_timestamp(input_file_path, output_file_path)
    print(result_message)'''

    file_path = "D://0_master//Lab//datasets//Historical Hourly Weather Data 2012-2017//humidity.csv"
    # Load the data
    data = pd.read_csv(file_path)
    for col_name in data.columns:
        # 处理空值
        if data[col_name].isnull().any():
            data[col_name].fillna(data[col_name].mean(), inplace=True)  # 用平均值填充


    # data = data.iloc[2:]

    ''' 
    data.reset_index(drop=True, inplace=True)

    # Convert the first column (time dimension) to datetime and then to Unix timestamp
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0]).astype('int64') // 10 ** 9  # Convert to seconds
    # Determine the number of dimensions and entries
    num_dimensions = data.shape[1]
    num_entries = data.shape[0]

    # Calculate the range for each dimension
    data_ranges = [(data[col].min(), data[col].max()) for col in data.columns]
    # Calculate the range for each dimension
    # data_ranges = [(data[col].min(), data[col].max()) if data[col].dtype != 'object'
    #              else ('N/A', 'N/A') for col in data.columns]
    print(data_ranges)

    # n_dim = 10
    num_queries = 100
    # num_entries = 10000

    # 生成数据集和数据范围
    # data_set, data_ranges = generate_data_set(num_entries, n_dim)

    # 生成查询
    queries = generate_n_dim_queries(num_queries, data_ranges)
    print(queries)

    # 执行查询并计时
    start_time = time.time()
    query_results = query_data_set(data, queries)
    end_time = time.time()

    for result in query_results:
        print(result)
    # 输出查询结果的一些统计信息
    query_result_lengths = [len(result) for result in query_results]
    print("每个查询返回的数据点数量:", query_result_lengths)
    print("查询时间 (秒):", end_time - start_time)
'''


