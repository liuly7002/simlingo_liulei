
import glob
import shutil
import pathlib


# 数据集目录
# dataset_path = '/home/liulei/ll/simlingo/database/simlingo_v2_2026_02_28/data/simlingo'
dataset_path = '/root/simlingo/database/simlingo_v2_2026_05_25/data/simlingo'


all_data_folders = glob.glob(f'{dataset_path}/**/Town*', recursive=True)
delete = True   # True: 真正的删除需要删除的文件夹 False: 打印需要删除的文件夹

# if multiple foulder only differ by the time and date (last part of the path) we delete all but the newest one
already_checked_root = []
num_deleted = 0          # 统计将要删除的目录数量
for data_folder in all_data_folders:
    data_folder = pathlib.Path(data_folder)   # 把字符串路径转为 Path 对象，方便操作 .name、.parent 等属性。
    data_folder_name = data_folder.name       # 获取当前路径的最后一级目录名
    data_folder_parts = data_folder_name.split('route')[-1]
    # remove everything before first _
    data_folder_date_time = '_'.join(data_folder_parts.split('_')[1:])
    path_without_date_time = str(data_folder).split(data_folder_date_time)[0]
    if path_without_date_time in already_checked_root:
        continue
    already_checked_root.append(path_without_date_time)
    
    all_data_folders_without_date_time = glob.glob(f'{path_without_date_time}*')
    if len(all_data_folders_without_date_time) > 1:
        all_data_folders_without_date_time.sort()
        for folder in all_data_folders_without_date_time[:-1]:
            print(f"Deleting {folder}")
            num_deleted += 1
            if delete:
                shutil.rmtree(folder) # uncomment to delete the folder

print(f"Deleted {num_deleted} folders out of {len(all_data_folders)}")