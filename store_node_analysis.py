import pandas as pd


def loadData():
    # Load the Excel file into a DataFrame
    input_file_10 = './0721_store_node/output_batch_size_100_query_size_20_node_num_10.xlsx'
    input_file_20 = './0721_store_node/output_batch_size_100_query_size_20_node_num_20.xlsx'
    input_file_40 = './0721_store_node/output_batch_size_100_query_size_20_node_num_40.xlsx'
    input_file_80 = './0721_store_node/output_batch_size_100_query_size_20_node_num_80.xlsx'
    input_file_160 = './0721_store_node/output_batch_size_100_query_size_20_node_num_160.xlsx'
    input_file_320 = './0721_store_node/output_batch_size_100_query_size_20_node_num_320.xlsx'
    df_10 = pd.read_excel(input_file_10)
    df_20 = pd.read_excel(input_file_20)
    df_40 = pd.read_excel(input_file_40)
    df_80 = pd.read_excel(input_file_80)
    df_160 = pd.read_excel(input_file_160)
    df_320 = pd.read_excel(input_file_320)
    return [df_10, df_20, df_40, df_80, df_160, df_320]


def strip_split(data_string):
    return data_string.strip('()').split(',')


def totally():
    file_path = './store_node_results/totally.xlsx'
    df = pd.read_excel(file_path)
    data_list = loadData()
    ratio_cut_time = data_list[0].at[0, "20"]
    lsh_time = 0.0207320173841156

    proposed_r_d_trans_time = {"10": 0, "20": 0, "40": 0, "80": 0, "160": 0, "320": 0}
    start = 10
    for index, data_df in enumerate(data_list):
        proposed_r_d_trans_time[str(start * (pow(2, index)))] = float(strip_split(data_df.at[4, "20"])[0])
    print(proposed_r_d_trans_time)

    basic_r_time = {"10": 0, "20": 0, "40": 0, "80": 0, "160": 0, "320": 0}
    start = 10
    for index, data_df in enumerate(data_list):
        basic_r_time[str(start * (pow(2, index)))] = float(strip_split(data_df.at[6, "20"])[0])
    print(basic_r_time)

    for column_name, column_data in df.items():
        print(column_name)
        kind = column_name.split("-")[0]
        store_node_num = column_name.split("-")[1]
        if kind == "TimeChain":
            # 计算查询
            df.at[0, column_name] = 0.347 + proposed_r_d_trans_time[store_node_num] * 2 + 6.33 + 0.00119209289550781 + lsh_time
            # 计算存储
            df.at[1, column_name] = ratio_cut_time + lsh_time * 2 + 6.33 + 0.00196 + proposed_r_d_trans_time[store_node_num] + 15.46
        if kind == "SEBDB":
            # 计算查询
            df.at[0, column_name] = 0.165 + basic_r_time[store_node_num] * 2 + 6.33 + 0.00119209289550781 + lsh_time
            # 计算存储
            df.at[1, column_name] = lsh_time * 2 + 6.33 + 0.00262 + basic_r_time[store_node_num] + 15.46
        if kind == "FileDES":
            # 计算查询
            df.at[0, column_name] = 10000 / 2 * 6.33 + basic_r_time[store_node_num] * 2 + 6.33 + 0.00119209289550781 + lsh_time
            # 计算存储
            df.at[1, column_name] = lsh_time * 2 + 6.33 + basic_r_time[store_node_num] + 15.46
    output_file_path = './store_node_results/totally.xlsx'
    df.to_excel(output_file_path, index=False)
    print(f"Modified Excel file has been saved to {output_file_path}")

    return 0


def main():
    totally()
    return 0


if __name__ == '__main__':
    main()