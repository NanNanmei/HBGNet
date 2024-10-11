import cv2
import os
import numpy as np
import sys
import glob

imaPath = r'E:\Change_detection\HRSCD\images_2012\2012\D14'         #输入需要处理的影像图片所在的文件夹D35
labelPath = r'E:\Change_detection\HRSCD\labels_land_cover_2006\2006\caijian'         #输入需要处理的标签图片所在的文件夹D35
s_g = 0.3         #比例因子o-1,小于这个比例就删掉

# image1 = []
# imageList1 = glob.glob(os.path.join(imaPath, '*.tif'))
# for item in imageList1:
#     image1.append(os.path.basename(item))
# image2 = []
# imageList2 = glob.glob(os.path.join(labelPath, '*.tif'))

#
# for item in imageList2:
#     image2.append(os.path.basename(item))

lablist= os.listdir(labelPath)
# imaList = os.listdir(imaPath)
for labels in lablist:

    label_path = os.path.join(labelPath,labels)

    img = cv2.imread(label_path, 0)
    s=img.size

    # 先利用二值化去除图片噪声
    ret, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
    area = 0
    height, width = img.shape
    yuzhi = s_g                   #比例大小
    yuzhi = s*yuzhi
    yuzhi = np.array(yuzhi, dtype='uint64')  # 转变为8字节型
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    if area <= yuzhi:
        os.remove(labelPath + r"\\" + labels)
    else:
        print('')

image1 = []
imageList1 = glob.glob(os.path.join(labelPath, '*.tif'))
for item in imageList1:
    image1.append(os.path.basename(item))

image2 = []
imageList2 = glob.glob(os.path.join(imaPath, '*.tif'))
for item in imageList2:
    image2.append(os.path.basename(item))

a = list(set(image2).difference(set(image1)))
print(a)
for x in a:
    os.remove(imaPath + r"\\" + x)