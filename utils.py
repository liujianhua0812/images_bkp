from sklearn.metrics import confusion_matrix

# 用于计算预测集和真实集的对比结果 评估目标检测模型的性能，主要通过计算 IoU（交并比）来判断检测框是否与真实框匹配，并利用混淆矩阵评估检测结果的准确性
def compute_iou(box1, box2):
    """ 计算 IoU(交并比) """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_results(detections, ground_truths, iou_thresh=0.3):
    """ 计算召回率和精确度 """
    y_true = []  # 真实值（1 = 目标存在, 0 = 目标不存在）
    y_pred = []  # 预测值（1 = 目标检测到, 0 = 目标未检测到）

    for gt_box in ground_truths:
        matched = False
        for det_box in detections:
            iou = compute_iou(gt_box, det_box)
            if iou >= iou_thresh:
                matched = True
                break
        y_true.append(1)
        y_pred.append(1 if matched else 0)

    for det_box in detections:
        matched = any(compute_iou(det_box, gt_box) >= iou_thresh for gt_box in ground_truths)
        if not matched:
            y_true.append(0)
            y_pred.append(1)
    return confusion_matrix(y_true, y_pred)


""" 
confusion_matrix(y_true, y_pred) 返回：
[[TN, FP],
 [FN, TP]]

TN (True Negative)：没有目标，模型也未检测到。
FP (False Positive)：没有目标，模型错误检测到。
FN (False Negative)：有目标，但模型未检测到（漏检）。
TP (True Positive)：有目标，模型正确检测到。

例如
detections = [(10, 10, 50, 50), (60, 60, 100, 100)]  # 预测框
ground_truths = [(15, 15, 55, 55), (80, 80, 120, 120)]  # 真实框

conf_matrix = evaluate_results(detections, ground_truths, iou_thresh=0.3)
print(conf_matrix)

真实框 (15,15,55,55) vs 预测框 (10,10,50,50)
IoU ≈ 0.56 ✅ 匹配
真实框 (80,80,120,120) vs 预测框 (60,60,100,100)
IoU ≈ 0.14 ❌ 未匹配
y_true = [1, 1, 0]（真实存在 2 个目标，1 个 FP）。
y_pred = [1, 0, 1]（1 个正确匹配，1 个 FN，1 个 FP）。\
[[0 1]  # 1 个 FP
 [1 1]] # 1 个 TP, 1 个 FN
 """