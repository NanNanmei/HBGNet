"""
The role of this file completes the data reading
"dist_mask" is obtained by using Euclidean distance transformation on the mask
"dist_contour" is obtained by using quasi-Euclidean distance transformation on the mask
"""

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile

from skimage import io
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import os
from osgeo import gdal
import tifffile as tiff
### Reading and saving of remote sensing images (Keep coordinate information)
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  波段数
    bands = dataset.RasterCount
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj


#保存遥感影像
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
          im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
           dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


#######
# class DatasetImageMaskContourDist(Dataset):
#
#     def __init__(self,  file_names):
#
#         self.file_names = file_names
#         # self.distance_type = distance_type
#         # self.dir = dir
#
#     def __len__(self):
#
#         return len(self.file_names)
#
#     def __getitem__(self, idx):
#
#         img_file_name = self.file_names[idx]
#         image = load_image(img_file_name)
#         mask = load_mask(img_file_name)
#         contour = load_contour(img_file_name)
#         # dist = load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)
#
#         return img_file_name, image, mask, contour

###训练的时候用这个
class DatasetImageMaskContourDist(Dataset):

    def __init__(self, dir, file_names):

        self.file_names = file_names
        # self.distance_type = distance_type
        self.dir = dir

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(os.path.join(self.dir,img_file_name+'.tif'))
        mask = load_mask(os.path.join(self.dir,img_file_name+'.tif'))
        contour = load_contour(os.path.join(self.dir,img_file_name+'.tif'))
        # dist = load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)
        dist = load_distance(os.path.join(self.dir, img_file_name+'.tif'))

        return img_file_name, image, mask, contour, dist




def load_image(path):

    img = Image.open(path)
    data_transforms = transforms.Compose(
        [
           # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )
    img = data_transforms(img)

    return img
#
#
# def load_mask(path):
#     # mask = cv2.imread(path.replace("image", "mask").replace("tif", "tif"), 0)
#     mask = cv2.imread(path.replace("train_image", "train_mask").replace("tif", "tif"), 0)
#    # im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(path.replace("image", "mask").replace("tif", "tif"))
#     mask = mask/255.
#     # mask[mask == 255] = 1
#     # mask[mask == 0] = 0
#
#     return torch.from_numpy(np.expand_dims(mask, 0)).float()

# def load_image(path):
#     img = tiff.imread(path).transpose([2, 0, 1]) #  array([3, 512, 512])
#     img = img.astype('uint8')
#     img = (img - img.min())/(img.max()-img.min())
#
#     return torch.from_numpy(img).float()




def load_mask(path):
    # mask = cv2.imread(path.replace("train_images", "train_labels").replace("tif", "tif"), 0)
    # im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(path.replace("train_image", "train_mask").replace("image", "label"))
    im_width, im_height, im_bands, mask, im_geotrans, im_proj = readTif(
        path.replace("train_image", "train_mask"))
    mask = mask.astype('uint8')
    mask = mask/255.
    # mask = np.reshape(mask, mask.shape + (1,))
    ###mask = mask/225.
    # mask[mask == 1] = 1
    # mask[mask == 0] = 0
    # print(mask.shape())
    # return torch.from_numpy(mask).long()

    return torch.from_numpy(np.expand_dims(mask, 0)).float()


def load_contour(path):

    # contour = cv2.imread(path.replace("train_image", "train_boundary").replace("tif", "tif"), 0)
    # contour = contour.astype('uint8')
    # im_width, im_height, im_bands, contour, im_geotrans, im_proj = readTif(path.replace("train_image", "train_boundary").replace("image", "label"))
    im_width, im_height, im_bands, contour, im_geotrans, im_proj = readTif(
        path.replace("train_image", "train_boundary"))
    contour = contour.astype('uint8')
    contour = contour/255.
    # contour[contour ==255] = 1
    # contour[contour == 0] = 0


    return torch.from_numpy(np.expand_dims(contour, 0)).long()

def load_distance(path):
    # im_width, im_height, im_bands, dist, im_geotrans, im_proj = readTif(path.replace("train_image", "train_dist").replace("image", "label"))
    im_width, im_height, im_bands, dist, im_geotrans, im_proj = readTif(
        path.replace("train_image", "train_dist"))
    dist = dist.astype('uint8')
    dist = dist/255.
    # dist = np.reshape(dist, dist.shape+(1,))
    return torch.from_numpy(np.expand_dims(dist, 0)).float()


