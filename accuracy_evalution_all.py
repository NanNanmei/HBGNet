import warnings
warnings.filterwarnings('ignore') # "error", "ignore", "always", "default", "module" or "once"
import glob
import os
import numpy as np
# import rasterio
from osgeo import gdal
import cv2
from scipy.ndimage import label
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from scipy import stats

def pixel_eval(pred_path, true_path):
    try:
        pred_img = gdal.Open(pred_path)
        y_pred = pred_img.ReadAsArray()

        actual_img = gdal.Open(true_path)
        y_true = actual_img.ReadAsArray()

        # 将tif影像转换为二进制掩码
        y_pred = (y_pred == 255).astype(np.int8)
        y_true = (y_true == 255).astype(np.int8)


        # 计算 TP, FP, FN, TN
        TP = np.sum((y_true == 1) & (y_pred == 1))  # 预测为正且实际也为正的像素数
        FP = np.sum((y_true == 0) & (y_pred == 1))  # 预测为正但实际为负的像素数
        FN = np.sum((y_true == 1) & (y_pred == 0))  # 预测为负但实际为正的像素数
        TN = np.sum((y_true == 0) & (y_pred == 0))  # 预测为负且实际也为负的像素数

        # precision = TP / (TP + FP + 1e-10)
        # recall = TP / (TP + FN + 1e-10)
        IoU = TP / (TP + FP + FN + 1e-10)
        acc = (TP + TN) / (TP + FP + FN + TN)
        f1 = 0


        # 假设 y_true 和 y_pred 已经准备好
        # precision = precision_score(y_true, y_pred, average='weighted')  # 对于多分类问题，可以指定'weighted'平均方式
        # recall = recall_score(y_true, y_pred, average='weighted')
        # f1 = f1_score(y_true, y_pred, average='weighted')

        # 如果是二分类问题，可以省略average参数或者设置为'micro'或'macro'
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        mcc = matthews_corrcoef(y_true.reshape(-1), y_pred.reshape(-1))


        if acc > 0.99 and TP == 0:
            precision = 1
            recall = 1
            IoU = 1

        if precision > 0 and recall > 0:
            f1 = f1_score(y_true, y_pred, average='micro')

        return acc, precision, recall, f1, mcc, IoU
    except Exception as e:
        return 0, 0, 0, 0, 0


def edge_eval(pred_path, true_path):
    try:
        # 读取预测的边缘和实际的边缘tif影像
        # with rasterio.open(pred_path) as pred_img:
        #     predicted_edges = pred_img.read(1)
        #
        # with rasterio.open(true_path) as actual_img:
        #     actual_edges = actual_img.read(1)

        pred_img = gdal.Open(pred_path)
        predicted_edges = pred_img.ReadAsArray()

        actual_img = gdal.Open(true_path)
        actual_edges = actual_img.ReadAsArray()

        # 将tif影像转换为二进制掩码
        predicted_edges = (predicted_edges == 255).astype(np.uint8)
        actual_edges = (actual_edges == 255).astype(np.uint8)

        # 定义膨胀结构元素，例如：一个3x3的单位矩阵（全1）
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # 对图像进行膨胀操作
        predicted_edges = cv2.dilate(predicted_edges, structuring_element)
        actual_edges = cv2.dilate(actual_edges, structuring_element)

        # 计算Completeness (Com)
        true_positive = np.sum(predicted_edges & actual_edges)
        completeness = true_positive / np.sum(actual_edges)



        # 计算Correctness (Corr)
        correctness = true_positive / np.sum(predicted_edges)


        # 计算F1-score (Fedge)
        fedge = 2 * (completeness * correctness) / (completeness + correctness)

        return completeness, correctness, fedge
    except Exception as e:
        return 0, 0, 0


def obejct_eval(pred_path, true_path):
    try:
        pred_img = gdal.Open(pred_path)
        y_pred = pred_img.ReadAsArray()

        actual_img = gdal.Open(true_path)
        y_true = actual_img.ReadAsArray()

        # 将tif影像转换为二进制掩码
        y_pred = (y_pred == 255).astype(np.int8)
        y_true = (y_true == 255).astype(np.int8)

        structure = np.ones((3, 3), dtype=int)
        labeled_M, num_M = label(y_pred, structure=structure)
        labeled_O, num_O = label(y_true, structure=structure)

        GOC, GUC, GTC = 0.0, 0.0, 0.0
        sum_area_Mi = 0.0
        for i in range(1, num_M + 1):
            Mi = labeled_M == i
            Oj_list = []
            Mi_n_Oj_list = []
            for j in range(1, num_O + 1):
                Oj = labeled_O == j
                Oj_list.append(Oj)
                Mi_n_Oj_list.append(np.sum(Mi & Oj))
            over_max_index = np.argmax(Mi_n_Oj_list)
            Oi = Oj_list[over_max_index]
            Mi_n_Oi = Mi_n_Oj_list[over_max_index]

            area_Mi_n_Oi = np.sum(Mi_n_Oi)
            area_Oi = np.sum(Oi)
            area_Mi = np.sum(Mi)

            sum_area_Mi += area_Mi

            OC = 1 - area_Mi_n_Oi / area_Oi
            UC = 1 - area_Mi_n_Oi / area_Mi

            TC = np.sqrt((OC ** 2 + UC ** 2) / 2)

            GOC += (OC * area_Mi)
            GUC += (UC * area_Mi)
            GTC += (TC * area_Mi)

        GOC /= sum_area_Mi
        GUC /= sum_area_Mi
        GTC /= sum_area_Mi
        return GOC, GUC, GTC

    except Exception as e:
        return 0, 0, 0


if __name__ == '__main__':
    # 这里放预测影像的mask路径
    root_path = r"D:\DeepL\data\NL\NL_test_lmx\E2EVAP\pre_mask"
    # pred_mask_tifs = glob.glob(os.path.join('./test_mask', '*.tif'))
    pred_mask_tifs = glob.glob(os.path.join(root_path, '*.tif'))

    list_precision = []
    list_acc = []
    list_recall = []
    list_f1 = []
    list_mcc = []

    list_IOU = []

    list_completeness = []
    list_correctness = []
    list_fedge = []
    list_GOC, list_GUC, list_GTC = [], [], []

    for pred_mask_tif in pred_mask_tifs:
        true_mask_tif = pred_mask_tif.replace('pre_mask', 'train_mask')

        pred_boundary_tif = pred_mask_tif.replace('pre_mask', 'pre_boundary')
        true_boundary_tif = pred_mask_tif.replace('pre_mask', 'train_boundary')#.replace('ortho', 'ortholabel')
        # precision, recall, f1, mcc = pixel_eval(r'test_mask/JS1_000255_000255.tif', r'mask/JS1_000255_000255.tif')

        # precision, recall, f1, mcc = pixel_eval(pred_mask_tif, true_mask_tif)
        acc, precision, recall, f1, mcc, IoU = pixel_eval(pred_mask_tif, true_mask_tif)
        list_acc.append(acc)
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1)
        list_mcc.append(mcc)
        list_IOU.append(IoU)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1 Score: ", f1)
        # print("Matthews Correlation Coefficient (MCC): ", mcc)

        # completeness, correctness, fedge = edge_eval(r'test_boundary/JS1_000255_000255.tif',
        #                                              r'boundary/JS1_000255_000255.tif')
        completeness, correctness, fedge = edge_eval(pred_boundary_tif, true_boundary_tif)
        # print("Completeness (Com):", completeness)
        # print("Correctness (Corr):", correctness)
        # print("F1-score (Fedge):", fedge)
        if (completeness != 0) and (not np.isnan(completeness)):
            list_completeness.append(completeness)
        if (correctness != 0) and (not np.isnan(correctness)):

            list_correctness.append(correctness)
            list_fedge.append(fedge)

        # GOC, GUC, GTC = obejct_eval(r'test_mask/JS1_000255_000255.tif', r'mask/JS1_000255_000255.tif')
        GOC, GUC, GTC = obejct_eval(pred_mask_tif, true_mask_tif)
        # print("GOC: ", GOC)
        # print("GUC: ", GUC)
        # print("GTC: ", GTC)
        list_GOC.append(GOC)
        list_GUC.append(GUC)
        list_GTC.append(GTC)

    print(f"overall accuracy: {np.mean(list_acc):.3f} ± {np.std(list_acc)}")
    print(f"Precision: {np.mean(list_precision):.3f} ± {np.std(list_precision)}")
    print("Recall: {:.3f} ± {:.3f}".format(np.mean(list_recall), np.std(list_recall)))
    print("F1 Score: {:.3f} ± {:.3f}".format(np.mean(list_f1), np.std(list_f1)))
    print("Matthews Correlation Coefficient (MCC): {:.3f} ± {:.3f}".format(np.mean(list_mcc), np.std(list_mcc)))

    print("IOU: {:.3f} ± {:.3f}".format(np.mean(list_IOU), np.std(list_IOU)))

    print("Completeness (Com): {:.3f} ± {:.3f}".format(np.mean(list_completeness), np.std(list_completeness)))
    print("Correctness (Corr): {:.3f} ± {:.3f}".format(np.mean(list_correctness), np.std(list_correctness)))
    print("F1-score (F bdy): {:.3f} ± {:.3f}".format(np.mean(list_fedge), np.std(list_fedge)))
    print("GOC: {:.3f} ± {:.3f}".format(np.mean(list_GOC), np.std(list_GOC)))
    print("GUC: {:.3f} ± {:.3f}".format(np.mean(list_GUC), np.std(list_GUC)))
    print("GTC: {:.3f} ± {:.3f}".format(np.mean(list_GTC), np.std(list_GTC)))


