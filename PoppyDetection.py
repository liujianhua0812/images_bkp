from ultralytics import YOLO
import os
import cv2
import numpy as np
from sahi.predict import predict, get_sliced_prediction, AutoDetectionModel
from utils import compute_iou, evaluate_results

def train_model(base_model_path, project_dir):
    model = YOLO(base_model_path)
    model.train(data=r'dataset/data.yaml',
                imgsz=640,
                epochs=500,
                batch=32,
                lr0 = 0.01,
                lrf=0.00001,
                workers=4,
                optimizer="SGD",
                seed = 0,
                device=[0,1],
                project=project_dir + 'train/',
                name='yingsu',
                patience = 100,
                val = True,
                scale=0.5,
                cache=False,
                multi_scale=True,
                weight_decay=0.0005,  # 防止过拟合
                augment=True,  # 启用数据增强
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # 颜色增强，适用于无人机拍摄不同光照环境
                perspective=0.0005,  # 透视变换，模拟无人机不同角度拍摄
                flipud=0.5, fliplr=0.5,  # 上下 & 左右翻转，增强泛化能力
                mosaic=1.0,  # 开启 mosaic 数据增强
                mixup=0.2
            )

def simple_test(best_model_path, project_dir):
    model = YOLO(best_model_path)
    model.val(data='dataset/data_test.yaml',
        batch = 1,
        imgsz = 4500,
        conf = 0.001,
        iou = 0.3,
        save_json=True,
        device=[0],
        project=project_dir + 'simple_test/'
    )

def sahi_test(best_model_path, project_dir, is_vis=False):
    sahi_slice_height = 1000
    sahi_slice_width = 1000
    sahi_conf_thres = 0.1
    sahi_score_thres = 0.1
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=best_model_path,
        confidence_threshold=sahi_conf_thres,
        device="cuda:0",  # or 'cuda:0'
    )
    test_images_folder = "dataset/train_and_val_and_test/images/test"
    test_labels_folder = "dataset/train_and_val_and_test/labels/test"
    # vis_folder =  project_dir + 'sahi_test/vis/' 

    # # 创建输出文件夹
    # if is_vis:
    #     os.makedirs(vis_folder, exist_ok=True)

    all_cm = np.array([[0, 0], [0, 0]])
    # 遍历图片文件夹
    for image_file in os.listdir(test_images_folder):
        print(image_file)
        if image_file.lower().endswith(('.jpg')):
            image_path = os.path.join(test_images_folder, image_file)
            label_path = os.path.join(test_labels_folder, os.path.splitext(image_file)[0] + ".txt")
            result = get_sliced_prediction(
                        image_path,
                        detection_model,
                        slice_height=1000,
                        slice_width=1000,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                    )
            # result.export_visuals(export_dir='./sahi_export', hide_labels=True, rect_th=3, file_name=image_file)
            # 读取图片
            image = cv2.imread(image_path)  
            height, width = image.shape[:2]
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            detected_boxes = [
                [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy] 
                for obj in result.object_prediction_list if obj.score.value > sahi_score_thres
            ]
            # 读取label文件
            gt_boxes = []
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
                            gt_boxes.append([x_min, y_min, x_max, y_max])
                        
                        else:
                            print(f"跳过格式错误的行: {label_path}")
                
                cm = evaluate_results(detected_boxes, gt_boxes)
                all_cm += cm
    precision = all_cm[1][1] / (all_cm[0][1] + all_cm[1][1])
    recall = all_cm[1][1] / (all_cm[1][0] + all_cm[1][1])
    print("Sahi: precisin: {0} recall: {1}".format(precision, recall))

if __name__ == '__main__':
    model_path = "yolo11/yolo11l.pt"
    project_path = "runs0/"
    mode = 0 # train
    mode = 1 # test
    if mode == 0:
        train_model(model_path, project_path)
    else:
        test_folder_id = 2
        best_model = project_path + 'train/yingsu{0}/weights/best.pt'.format(test_folder_id)
        simple_test(best_model, project_path)
        sahi_test(best_model, project_path)


    