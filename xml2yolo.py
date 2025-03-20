import xml.etree.ElementTree as ET
import os, cv2
import numpy as np
from os import listdir
from os.path import join

classes = []
# XML转化为yolo所需的txt格式文件
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlpath, xmlname):
    with open(xmlpath, "r", encoding='utf-8') as in_file:
        txtname = xmlname[:-4] + '.txt'
        txtfile = os.path.join(txtpath, txtname)
        tree = ET.parse(in_file)
        root = tree.getroot()
        filename = root.find('filename')
        img = cv2.imdecode(np.fromfile('{}/{}.{}'.format(imgpath, xmlname[:-4], postfix), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        res = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            res.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
        if len(res) != 0:
            with open(txtfile, 'w+') as f:
                f.write('\n'.join(res))


if __name__ == "__main__":
    postfix = 'JPG'    # 图像后缀
    imgpath = r'/home/public/yjl/yolo/dataset/test/images'    # 图像文件路径
    xmlpath = r'/home/public/yjl/yolo/dataset/test/labels'   # xml文件文件路径
    txtpath = r'/home/public/yjl/yolo/dataset/test/labels_yolo'      # 生成的txt文件路径
    
    if not os.path.exists(txtpath):
        os.makedirs(txtpath, exist_ok=True)
    
    list = os.listdir(xmlpath)
    error_file_list = []
    for i in range(0, len(list)):
        try:
            path = os.path.join(xmlpath, list[i])
            if ('.xml' in path) or ('.XML' in path):
                convert_annotation(path, list[i])
                print(f'file {list[i]} convert success.')
            else:
                print(f'file {list[i]} is not xml format.')
        except Exception as e:
            print(f'file {list[i]} convert error.')
            print(f'error message:\n{e}')
            error_file_list.append(list[i])
    print(f'those file convert failure: {error_file_list}')
    print(f'Dataset Classes:{classes}')


# import os
# import xml.etree.ElementTree as ET
 
# # 定义类别顺序
# categories = ['yingsu']
# category_to_index = {category: index for index, category in enumerate(categories)}
 
# # 定义输入文件夹和输出文件夹
# input_folder = r'/home/public/yjl/dataset/train/labels'  # 替换为实际的XML文件夹路径
# output_folder = r'/home/public/yjl/dataset/train/labels_yolo'  # 替换为实际的输出TXT文件夹路径
 
# # 确保输出文件夹存在
# os.makedirs(output_folder, exist_ok=True)
 
# # 遍历输入文件夹中的所有XML文件
# for filename in os.listdir(input_folder):
#     if filename.endswith('.xml'):
#         xml_path = os.path.join(input_folder, filename)
#         # 解析XML文件
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         # 提取图像的尺寸
#         size = root.find('size')
#         width = int(size.find('width').text)
#         height = int(size.find('height').text)
#         # 存储name和对应的归一化坐标
#         objects = []
 
#         # 遍历XML中的object标签
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             if name in category_to_index:
#                 category_index = category_to_index[name]
#             else:
#                 continue  # 如果name不在指定类别中，跳过该object
 
#             bndbox = obj.find('bndbox')
#             xmin = int(bndbox.find('xmin').text)
#             ymin = int(bndbox.find('ymin').text)
#             xmax = int(bndbox.find('xmax').text)
#             ymax = int(bndbox.find('ymax').text)
 
#             # 转换为中心点坐标和宽高
#             x_center = (xmin + xmax) / 2.0
#             y_center = (ymin + ymax) / 2.0
#             w = xmax - xmin
#             h = ymax - ymin
 
#             # 归一化
#             x = x_center / width
#             y = y_center / height
#             w = w / width
#             h = h / height
 
#             objects.append(f"{category_index} {x} {y} {w} {h}")
 
#         # 输出结果到对应的TXT文件
#         txt_filename = os.path.splitext(filename)[0] + '.txt'
#         txt_path = os.path.join(output_folder, txt_filename)
#         print(txt_path)
#         with open(txt_path, 'w') as f:
#             for obj in objects:
#                 f.write(obj + '\n')

