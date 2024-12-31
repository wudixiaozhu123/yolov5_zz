import os

labels_dir = 'C:\\yolov5-master\\data\\dataset\\cifar_cate\\train\\labels'  # 设置您的标签文件夹路径
num_classes = 3  # 替换成你实际的类别数

for label_file in os.listdir(labels_dir):
    label_file_path = os.path.join(labels_dir, label_file)
    
    # 打印每个标签文件的内容
    with open(label_file_path, 'r') as f:
        print(f"Contents of {label_file}:")
        for line in f:
            print(line.strip())  # 打印每行内容
