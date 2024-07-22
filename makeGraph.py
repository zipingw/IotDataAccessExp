import pandas as pd
def loadData(input_file):
    # Load the Excel file into a DataFrame
    # input_file = './0716_distance_batch_1/output_batch_size_100_query_points_10_0716_distance_batch_1.xlsx'
    df = pd.read_excel(input_file)
    return df


def strip_split(data_string):
    return data_string.strip('()').split(',')

def data_preprocess():
    data = loadData()
    print(data)

    for column in data.columns:
        # ratio_cut_time_1 = data.at[0, column]
        batch_nums_1 = data.at[1, column]


def consensus():
    map = {0: "1", 1: "10", 2: "20", 3: "25", 4: "40", 5: "50", 6: "80", 7: "100"}

    file_path = 'basic_results/consensus.xlsx'
    df = pd.read_excel(file_path)
    data = loadData()
    print(data)
    for index, row in df.iterrows():
        if pd.isna(row['tTimeChain']):
            df.at[index, 'tTimeChain'] = strip_split(data.at[4, map[int(index)]])[0]
        if pd.isna(row['tFileDES']):
            df.at[index, 'tFileDES'] = strip_split(data.at[7, map[int(index)]])[0]
        if pd.isna(row['tCRUSH']):
            df.at[index, 'tCRUSH'] = strip_split(data.at[10, map[int(index)]])[0]
        if pd.isna(row['fTimeChain']):
            df.at[index, 'fTimeChain'] = strip_split(data.at[4, map[int(index)]])[1]
        if pd.isna(row['fFileDES']):
            df.at[index, 'fFileDES'] = strip_split(data.at[7, map[int(index)]])[1]
        if pd.isna(row['fCRUSH']):
            df.at[index, 'fCRUSH'] = strip_split(data.at[10, map[int(index)]])[1]
        if pd.isna(row['sTimeChain']):
            df.at[index, 'sTimeChain'] = strip_split(data.at[4, map[int(index)]])[2]
        if pd.isna(row['sFileDES']):
            df.at[index, 'sFileDES'] = strip_split(data.at[7, map[int(index)]])[2]
        if pd.isna(row['sCRUSH']):
            df.at[index, 'sCRUSH'] = strip_split(data.at[10, map[int(index)]])[2]
        if pd.isna(row['pTimeChain']):
            df.at[index, 'pTimeChain'] = strip_split(data.at[4, map[int(index)]])[3]
        if pd.isna(row['pFileDES']):
            df.at[index, 'pFileDES'] = strip_split(data.at[7, map[int(index)]])[3]
        if pd.isna(row['pCRUSH']):
            df.at[index, 'pCRUSH'] = strip_split(data.at[10, map[int(index)]])[3]

    output_file_path = 'basic_results/consensus.xlsx'
    df.to_excel(output_file_path, index=False)

    print(f"Modified Excel file has been saved to {output_file_path}")
    return 0


def ratio_cut():
    map = {0: "1", 1: "10", 2: "20", 3: "25", 4: "40", 5: "50", 6: "80", 7: "100"}

    file_path = 'basic_results/ratio_cut.xlsx'
    df = pd.read_excel(file_path)
    data = loadData()
    print(data)
    for index, row in df.iterrows():
        if pd.isna(row['tTimeChain']):
            df.at[index, 'tTimeChain'] = strip_split(data.at[4, map[int(index)]])[0]
        if pd.isna(row['tSEBDB']):
            df.at[index, 'tSEBDB'] = strip_split(data.at[3, map[int(index)]])[0]

        if pd.isna(row['bTimeChain']):
            df.at[index, 'bTimeChain'] = strip_split(data.at[1, map[int(index)]])[1]
        if pd.isna(row['bSEBDB']):
            df.at[index, 'bSEBDB'] = strip_split(data.at[1, map[int(index)]])[0]

        if pd.isna(row['nTimeChain']):
            df.at[index, 'nTimeChain'] = strip_split(data.at[2, map[int(index)]])[1]
        if pd.isna(row['nSEBDB']):
            df.at[index, 'nSEBDB'] = strip_split(data.at[2, map[int(index)]])[0]

        if pd.isna(row['RatioCut']):
            df.at[index, 'RatioCut'] = data.at[0, map[int(index)]]

    output_file_path = 'basic_results/ratio_cut.xlsx'
    df.to_excel(output_file_path, index=False)

    print(f"Modified Excel file has been saved to {output_file_path}")
    return 0


def totally():
    file_path = 'basic_results/totally.xlsx'
    df = pd.read_excel(file_path)
    input_file_path = './0716_distance_batch_1/output_batch_size_100_query_points_10_0716_distance_batch_1.xlsx'
    data = loadData(input_file_path)
    print(data)

    ratio_cut_time = {"1": 0, "10": 0, "20": 0, "25": 0, "40": 0, "50": 0, "80": 0, "100": 0}
    ratio_cut = data.iloc[0]
    for column_name, value in ratio_cut.items():
        ratio_cut_time[column_name] = value
    print(ratio_cut_time)

    lsh_time = {"1": 0.000321168271410796, "10": 0.0107538196256217, "20": 0.0207320173841156, "25": 0.0264767997794681,
                "40": 0.0407103305392795, "50": 0.0516250451405843, "80": 0.0810651690871627, "100": 0.101649188995361}
    proposed_r_d_trans_time = {"1": 0, "10": 0, "20": 0, "25": 0, "40": 0, "50": 0, "80": 0, "100": 0}
    proposed_r_d = data.iloc[4]
    for column_name, value in proposed_r_d.items():
        proposed_r_d_trans_time[column_name] = float(strip_split(value)[0])
    print(proposed_r_d_trans_time)

    basic_r_time = {"1": 0, "10": 0, "20": 0, "25": 0, "40": 0, "50": 0, "80": 0, "100": 0}
    basic_r = data.iloc[6]
    for column_name, value in basic_r.items():
        basic_r_time[column_name] = float(strip_split(value)[0])
    print(basic_r_time)
    print(df)

    for column_name, column_data in df.items():
        print(column_name)
        kind = column_name.split("-")[0]
        batch_size = column_name.split("-")[1]
        if kind == "TimeChain":
            # 计算查询
            df.at[0, column_name] = 0.347 + proposed_r_d_trans_time[batch_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time[batch_size]
            # 计算存储
            df.at[1, column_name] = ratio_cut_time[batch_size] + lsh_time[batch_size] * 2 + 6.33 + 0.00196 + proposed_r_d_trans_time[batch_size] + 15.46
        if kind == "SEBDB":
            # 计算查询
            df.at[0, column_name] = 0.165 + basic_r_time[batch_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time[batch_size]
            # 计算存储
            df.at[1, column_name] = lsh_time[batch_size] * 2 + 6.33 + 0.00262 + basic_r_time[batch_size] + 15.46
        if kind == "FileDES":
            # 计算查询
            df.at[0, column_name] = 10000 / 2 * 6.33 + basic_r_time[batch_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time[batch_size]
            # 计算存储
            df.at[1, column_name] = lsh_time[batch_size] * 2 + 6.33 + basic_r_time[batch_size] + 15.46
    output_file_path = 'basic_results/totally.xlsx'
    df.to_excel(output_file_path, index=False)
    print(f"Modified Excel file has been saved to {output_file_path}")

    file_path2 = 'basic_results/device.xlsx'
    df2 = pd.read_excel(file_path2)
    map = {0: "1", 1: "10", 2: "20", 3: "25", 4: "40", 5: "50", 6: "80", 7: "100"}
    for index, row in df2.iterrows():
        print(index)
        if pd.isna(row['tTimeChain']):
            df2.at[index, 'tTimeChain'] = df.at[0, 'TimeChain-' + map[int(index)]] + df.at[1, 'TimeChain-' + map[int(index)]]
        if pd.isna(row['tFileDES']):
            df2.at[index, 'tFileDES'] = df.at[0, 'FileDES-' + map[int(index)]] + df.at[1, 'FileDES-' + map[int(index)]]
        if pd.isna(row['tSEBDB']):
            df2.at[index, 'tSEBDB'] = df.at[0, 'SEBDB-' + map[int(index)]] + df.at[1, 'SEBDB-' + map[int(index)]]
    for index, row in df2.iterrows():
        if pd.isna(row['dTimeChain']):
            df2.at[index, 'dTimeChain'] = 1000 * int(map[int(index)]) / df2.at[index, 'tTimeChain']
        if pd.isna(row['dFileDES']):
            df2.at[index, 'dFileDES'] = 1000 * int(map[int(index)]) / df2.at[index, 'tFileDES']
        if pd.isna(row['dSEBDB']):
            df2.at[index, 'dSEBDB'] = 1000 * int(map[int(index)]) / df2.at[index, 'tSEBDB']
    output_file_path_2 = 'basic_results/device.xlsx'
    df2.to_excel(output_file_path_2, index=False)
    print(f"Modified Excel file has been saved to {output_file_path_2}")
    return 0


def main():
    # data_preprocess()
    # consensus()
    # ratio_cut()
    totally()
    return 0

if __name__ == '__main__':
    main()