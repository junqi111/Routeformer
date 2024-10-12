import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """
    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    geo_file_path = args.geo_file_path
    grid_file_path = args.grid_file_path
    norm_each_channel = args.norm_each_channel
    steps_per_day = args.steps_per_day
    if_rescale = not norm_each_channel  # if evaluate on rescaled data

    # 读取 CHIBike.geo 文件
    geo_df = pd.read_csv(geo_file_path)

    # 读取 CHIBike.grid 文件
    grid_df = pd.read_csv(grid_file_path)
    grid_df['time'] = pd.to_datetime(grid_df['time'])

    # # # 过滤流量值低于5的样本
    # grid_df = grid_df[(grid_df['inflow'] >= 5) | (grid_df['outflow'] >= 5)]
    # # 检查网格单元数是否匹配
    # num_grid_cells = grid_df[['row_id', 'column_id']].drop_duplicates().shape[0]
    # print(f"唯一网格单元数量: {num_grid_cells}")

    # 将 inflow 和 outflow 作为特征，转换为三维数组
    grid_pivot = grid_df.pivot_table(index='time', columns=['row_id', 'column_id'], values=['inflow', 'outflow'])
    data = grid_pivot.values.reshape(len(grid_pivot), len(geo_df), -1)

    data = data[..., target_channel]
    ''''''
    print("raw time series shape: {0}".format(data.shape))

    # 数据集划分
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num:train_num + valid_num]
    test_index = index_list[train_num + valid_num:train_num + valid_num + test_num]

    # 数据标准化
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # 添加时间特征
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 / 7 for i in range(data_norm.shape[0])]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    # if add_day_of_month:
    #     dom = (grid_df['time'].dt.day - 1) / 31
    #     dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
    #     feature_list.append(dom_tiled)

    # if add_day_of_year:
    #     doy = (grid_df['time'].dt.dayofyear - 1) / 366
    #     doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
    #     feature_list.append(doy_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # 保存数据
    index = {"train": train_index, "valid": valid_index, "test": test_index}
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {"processed_data": processed_data}
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)

    # 生成邻接矩阵并保存
    adj_mx = np.zeros((len(geo_df), len(geo_df)))
    for i in range(len(geo_df)):
        adj_mx[i][i] = 1  # 自己与自己连接

    with open(output_dir + "/adj_mx.pkl", "wb") as f:
        pickle.dump(adj_mx, f)


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 6
    FUTURE_SEQ_LEN = 1
    STEPS_PER_DAY=48
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]  # target channel(s)

    DATASET_NAME = "CHIBike"
    TOD = True  # if add time_of_day feature
    DOW = True  # if add day_of_week feature
    DOM = True  # if add day_of_month feature
    DOY = True  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    GEO_FILE_PATH = "datasets/raw_data/{0}/{0}.geo".format(DATASET_NAME)
    GRID_FILE_PATH = "datasets/raw_data/{0}/{0}.grid".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--geo_file_path", type=str, default=GEO_FILE_PATH, help="Raw geo data file.")
    parser.add_argument("--grid_file_path", type=str, default=GRID_FILE_PATH, help="Raw grid data file.")
    parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN, help="History sequence length.")
    parser.add_argument("--future_seq_len", type=int, default=FUTURE_SEQ_LEN, help="Future sequence length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD, help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW, help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM, help="Add feature day_of_month.")
    parser.add_argument("--doy", type=bool, default=DOY, help="Add feature day_of_year.")
    parser.add_argument("--target_channel", type=list, default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO, help="Train ratio.")
    parser.add_argument("--valid_ratio", type=float, default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Whether to normalize each channel.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
