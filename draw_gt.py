import os
import cv2

# 根据原有imgs以及对应labels绘制标准预测结果，用于后续对比
# 设置文件夹路径
data_type = "train"
folder = "gt_train"
images_folder = "dataset/train_and_val_and_test/images/{0}".format(data_type)  # 替换为实际图片文件夹路径
labels_folder = "dataset/train_and_val_and_test/labels/{0}".format(data_type)  # 替换为实际标签文件夹路径
output_folder = "dataset/train_and_val_and_test/images/{0}".format(folder)  # 替换为输出文件夹路径

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历图片文件夹
for image_file in os.listdir(images_folder):
    print(image_file)
    if image_file.lower().endswith(('.jpg')):
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, os.path.splitext(image_file)[0] + ".txt")

        # 读取图片获取图片的宽高信息
        image = cv2.imread(image_path)  
        height, width = image.shape[:2]
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 读取label文件
        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    coords = list(map(float, line.strip().split()))
                    if len(coords) == 5:
                        idx, center_x, center_y, w, h = coords
                        center_x = center_x * width
                        w = w * width
                        center_y = center_y * height
                        h = h * height
                        x_min = int(center_x - w / 2)
                        x_max = int(center_x + w /2)
                        y_min = int(center_y - h / 2)
                        y_max = int(center_y + h / 2) 
                        # 画框 (红色, 线宽 2)
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 8)
                    else:
                        print(f"跳过格式错误的行: {label_path}")

        # 保存绘制后的图片
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)
        print(f"已处理并保存: {output_path}")

print("所有图片处理完成！")