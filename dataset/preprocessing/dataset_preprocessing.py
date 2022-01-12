import os
import numpy as np
import pandas as pd
import csv

def get_angle_data_frames(filepath: str , viewport_angle: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    angle_frames =  df.where(df[' Predicted_viewpoint'] == viewport_angle)
    return angle_frames[angle_frames[" Image_name"].notna()]

def get_production_year(image_name: str) -> str:
    return image_name.split("$$")[2]

def get_path_from_root(file_name: str) -> str:
    retv = file_name.split('$$')[:4]
    return os.path.join(*retv)

if __name__ == '__main__':
    print("Preprocessing dataset")

    dataset_root = '../../resized_DVM'

    csv_file = '/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_0.csv'
    # desired_angle = 45
    # confirmed_train_csv_file = f'../confirmed_angle_train.csv'
    # confirmed_val_csv_file = f'../confirmed_angle_val.csv'
    # confirmed_test_csv_file = f'../confirmed_angle_test.csv'
    # train_percentage = .8
    # val_percentage = .1
    # test_percentage = .1
    df = pd.read_csv(csv_file)
    # df = df[df["prod_year"] > 2000]
    # df = df[df["prod_year"] < 2019]
    df = df.sample(frac=1)
    split_points = (
        int(len(df) * 0.8),
        int(len(df) * (0.9))
    )
    train = df.iloc[:split_points[0]]
    val = df.iloc[split_points[0]:split_points[1]]
    test = df.iloc[split_points[1]:]
    train.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_train_0.csv', index=False)
    val.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_val_0.csv', index=False)
    test.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_test_0.csv', index=False)


    # df = get_angle_data_frames(csv_file, desired_angle)

    # data = []

    # print(f"Found {df.shape[0]} samples of a given angle")

    # for idx, row in df.iterrows():
    #     file_name = row[" Image_name"]
    #     prod_year = get_production_year(file_name)
    #     path_from_root = get_path_from_root(file_name)
    #     relative_path = os.path.join(path_from_root, file_name)
    #     data.append((relative_path, prod_year))

    # np.random.shuffle(data)
    # split_points = (
    #     int(len(data) * train_percentage),
    #     int(len(data) * (train_percentage + val_percentage))
    # )
    # train_data = data[:split_points[0]]
    # val_data = data[split_points[0]:split_points[1]]
    # test_data = data[split_points[1]:]
    # print(f'Train samples: {len(train_data)}, validation samples: {len(val_data)}, test samples: {len(test_data)}')

    # output_tuples = [
    #     (confirmed_train_csv_file, train_data),
    #     (confirmed_val_csv_file, val_data),
    #     (confirmed_test_csv_file, test_data)
    # ]

    # for file, data in output_tuples:
    #     with open(file, "w", newline='') as out:
    #         csv_out = csv.writer(out)
    #         csv_out.writerow(['image_path', 'prod_year'])
    #         csv_out.writerows(data)


    # file_path = os.path.join(dataset_root, data[564][0])

    # assert os.path.exists(file_path), f"File path: {file_path} does not exist"