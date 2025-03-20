from ultralytics import YOLO
model = YOLO('yolo11x.pt')
dataset = model.load_dataset('dataset/data.yaml')
print(f"训练集样本数量: {len(dataset)}")