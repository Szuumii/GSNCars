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

    dataset_root = '.../resized_DVM'

    csv_file = '~/Datasets/Image_table.csv'
    desired_angle = 45
    processed_csv_file = f'../processed_angle_{desired_angle}.csv'

    df = get_angle_data_frames(csv_file, desired_angle)

    data = []

    print(f"Found {df.size} samples of a given angle")

    for idx, row in df.iterrows():
        file_name = row[" Image_name"]
        prod_year = get_production_year(file_name)
        path_from_root = get_path_from_root(file_name)
        relative_path = os.path.join(path_from_root, file_name)
        data.append((relative_path, prod_year))

    with open(processes_csv_file, "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image_path', 'prod_year'])
        csv_out.writerows(data)


    file_path = os.path.join(dataset_root, data[564][0])

    assert os.path.exists(file_path), f"File path: {file_path} does not exist"