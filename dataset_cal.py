import pandas as pd

def cal_dataset():
    # query size 10
    # query size 20
    # query size 40
    lsh_time = 0.02073201
    proposed_r_d_trans_time = [8.72875, 9.23488, 9.240669]
    basic_r_time = [78.51747, 81.32912, 85.9776]
    ratio_cut_time = [17.3068, 17.5629, 17.4637]

    for query_size in range(3):
        print(f"query_size: {query_size}")
        # 计算查询

        timechain_query_time = 0.347 + proposed_r_d_trans_time[query_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time
        # 计算存储
        timechain_store_time = ratio_cut_time[query_size] + lsh_time * 2 + 6.33 + 0.00196 + proposed_r_d_trans_time[query_size] + 15.46
        print(f"timechain_query_time: {timechain_query_time}, timechain_store_time: {timechain_store_time}")

        # 计算查询
        SEBDB_query_time = 0.165 + basic_r_time[query_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time
        # 计算存储
        SEBDB_store_time = lsh_time * 2 + 6.33 + 0.00262 + basic_r_time[query_size] + 15.46
        print(f"SEBDB_query_time: {SEBDB_query_time}, SEBDB_store_time: {SEBDB_store_time}")

        # 计算查询
        FileDES_query_time = 10000 / 2 * 6.33 + basic_r_time[query_size] * 2 + 6.33 + 0.00119209289550781 + lsh_time
        # 计算存储
        FileDES_store_time = lsh_time * 2 + 6.33 + basic_r_time[query_size] + 15.46
        print(f"FileDES_query_time: {FileDES_query_time}, fileDES_store_time: {FileDES_store_time}")
    return 0


def main():

    cal_dataset()
    return 0


if __name__ == '__main__':
    main()