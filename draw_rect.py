import json
import cv2
import os

# 读取 JSON 格式的目标检测预测结果，然后在原始图片上绘制检测框 (bbox)，并将处理后的图片保存到 output_folder 中
# 配置路径
json_path = "val0/val37/predictions.json"  # JSON 文件路径
image_folder = "dataset/train_and_val_and_test/images/test"  # 存放原始图片的文件夹
output_folder = "dataset/train_and_val_and_test/images/pred2"  # 存放处理后图片的文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 读取 JSON 数据
with open(json_path, "r") as f:
    data = json.load(f)

# 处理每个 bbox
images = {}
for item in data:
    image_id = item["image_id"]
    bbox = {'bbox': item["bbox"], 'score': item['score']}
    if image_id in images:
        images[image_id].append(bbox)
    else:
        images[image_id] = [bbox]

for image_id, bboxes in images.items():
    for bbox in bboxes:
    
        # 计算 bbox 坐标（左上角 x, y 和宽高）
        x, y, w, h = map(int, bbox['bbox'])

        # 构造图片路径
        image_path = os.path.join(image_folder, f"{image_id}.JPG")
        save_path = os.path.join(output_folder, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        # 检查文件是否存在
        if os.path.exists(save_path):
            image = cv2.imread(save_path)
    

        # 画出 bbox（红色，粗 2 像素）
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 9)

        # 添加文本标签（类别 ID 和置信度）
        label = f"Score: {bbox['score']:.2f}"
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 9)

    # 生成输出路径
    output_path = os.path.join(output_folder, f"{image_id}.jpg")

    # 保存处理后的图像
    cv2.imwrite(output_path, image)

    print(f"Processed and saved: {output_path}")

print("All images processed.")