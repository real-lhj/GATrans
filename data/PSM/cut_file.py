import pandas as pd
from pathlib import Path
############################################
# @ res_file_path 待分割文件路径
res_file_path = Path(r"test_label.csv")
# @ split_size 分割大小 这里是10万条一个文件
split_size = 20000
############################################
tar_dir = res_file_path.parent/("split_"+res_file_path.name.split(".")[0])
if not tar_dir.exists():
    tar_dir.mkdir()
    print("创建文件夹\t"+str(tar_dir))
print("目标路径：\t"+str(tar_dir))
print("分割文件：\t"+str(res_file_path))
print("分割大小：\t"+"{:,}".format(split_size))
tmp = pd.read_csv(res_file_path)
print(res_file_path)
columns = tmp.columns.to_list()
idx = 0 
while(len(tmp)>0):
    start = 1+(idx*split_size)
    tmp = pd.read_csv(res_file_path,
                header = None,
                names = columns,
                skiprows = start,
                nrows = split_size)
    if len(tmp) <= 0:
        break
    file_name = res_file_path.name.split(".")[0]+"_{}_{}".format(start,start+len(tmp))+".csv"
    file_path = tar_dir/file_name
    tmp.to_csv(file_path,index=False)
    idx+=1
    print(file_name +"\t保存成功")
