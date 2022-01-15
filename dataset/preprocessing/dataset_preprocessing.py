import os
import numpy as np
import pandas as pd
import csv

def get_angle_data_frames(filepath: str , viewport_angle: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    angle_frames =  df.where(df[' Predicted_viewpoint'] == viewport_angle)
    return angle_frames[angle_frames[" Image_name"].notna()]

def get_production_year(image_name: str) -> str:
    return int(image_name.split("$$")[2])

def get_path_from_root(file_name: str) -> str:
    retv = file_name.split('$$')[:4]
    return os.path.join(*retv)

if __name__ == '__main__':
    print("Preprocessing dataset")

    dataset_root = "/home/shades/Datasets/resized_DVM/"

    csv_file = '/home/shades/GitRepos/GSNCars/csv_files/angle_45/processed_angle_45.csv'
    desired_angle = 45
    confirmed_train_csv_file = '/home/shades/GitRepos/GSNCars/csv_files/angle_45/processed_angle_45_train.csv'
    confirmed_val_csv_file = '/home/shades/GitRepos/GSNCars/csv_files/angle_45/processed_angle_45_val.csv'
    confirmed_test_csv_file = '/home/shades/GitRepos/GSNCars/csv_files/angle_45/processed_angle_45_test.csv'
    # train_percentage = .8
    # val_percentage = .1
    # test_percentage = .1
    angle = pd.read_csv(csv_file)

    data = []

    for idx, row in angle.iterrows():
        file_name = row[" Image_name"]
        prod_year = get_production_year(file_name)
        path_from_root = get_path_from_root(file_name)
        relative_path = os.path.join(path_from_root, file_name)
        data.append((relative_path, prod_year))

    # with open(out_csv, "w") as out:
    #     writer = csv.writer(out)
    #     writer.writerow(['image_path', 'prod_year'])
    #     for tuple in data:
    #         writer.writerows(data)
    

    df = pd.DataFrame(data, columns=['image_path', 'prod_year'])

    df = df[df["prod_year"] > 2000]
    df = df[df["prod_year"] < 2019]
    df = df.sample(frac=1)

    split_points = (
        int(len(df) * 0.8),
        int(len(df) * (0.9))
    )
    train = df.iloc[:split_points[0]]
    val = df.iloc[split_points[0]:split_points[1]]
    test = df.iloc[split_points[1]:]
    train.to_csv(confirmed_train_csv_file, index=False)
    val.to_csv(confirmed_val_csv_file, index=False)
    test.to_csv(confirmed_test_csv_file, index=False)

    # print('Creating regression dataset')

    # YEARS_BUCKETS = 9  # group years in a buckets of two
    # MAX_REGRESION_VALUE = 10  # max value for regression
    # MIN_YEAR = 2001
    # MAX_YEAR = 2018
    # BUCKET_REGRESION_STEP = MAX_REGRESION_VALUE / (YEARS_BUCKETS - 1)
    # YEARS = [i for i in range(MIN_YEAR, MAX_YEAR + 1)]
    # regression_year_map = {}

    # for year in YEARS:
    #     mapped_value = (year - MIN_YEAR) // 2  # For each 2 years assign same index from 0 to 8
    #     mapped_value = mapped_value * BUCKET_REGRESION_STEP
    #     regression_year_map[year] = mapped_value

    # df = pd.read_csv(csv_file)
    # df['prod_year'].replace(regression_year_map, inplace=True)
    # df = df.sample(frac=1)
    # df.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_0_reg.csv')

    # train = df.iloc[:split_points[0]]
    # val = df.iloc[split_points[0]:split_points[1]]
    # test = df.iloc[split_points[1]:]
    # train.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_train_0_reg.csv', index=False)
    # val.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_val_0_reg.csv', index=False)
    # test.to_csv('/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_test_0_reg.csv', index=False)


    # df = get_angle_data_frames(csv_file, desired_angle)

    # data = []

    # print(f"Found {df.shape[0]} samples of a given angle")

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