# Readme
## query_points.py
- 该文件引入query size为变量，以batch size和query size为两个变量跑了个双重循环得到实验结果放在以日期命名的文件夹中
- 目前最有效的实验结果放在文件夹"0716_distance_batch_1"中
## makeGraph.py
- 该文件将query_points.py中得到的实验数据统一处理并填充到excel表格中
## measurement文件夹
- 作图代码以及实验结果图
## ablation_study
- 基于query_points.py中的代码进行修改
- dataset_query_size_ablation_study.py 对不同数据集 使用不同的 query size 得到实验结果，并用makeGraph.py得到三个数据集下的查询与存储总时延
- store_node_num_ablation_study.py 对不同数量的store node进行消融实验分析