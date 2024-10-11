import torch
import os
from torch.utils.data import DataLoader
# from dataset import DatasetImageMaskContourDist
from dataset_RE import DatasetImageMaskContourDist
import glob
# from models import BsiNet,Field
# from model_Field1 import Field
from featuremap_Field import Field
# from models_Baseline import Field
# from model_noEdgeBrunch import Field
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
from torch import nn

def build_model(model_type):

    if model_type == "field":
        model = Field(num_classes=2)

    return model


if __name__ == "__main__":
    args = create_validation_arg_parser().parse_args()
    args.model_file = r"E:\zhaohang\DeepL\模型训练pt文件_lmx机子\JS_zh\80.pt"
    args.save_path = r"D:\DeepL\data\paper\JS\test\特征图可视化\pre_mask1"
    args.model_type = 'field'
    args.test_path = r"D:\DeepL\data\paper\JS\test\特征图可视化\train_image"

    # args = create_validation_arg_parser().parse_args()
    # args.model_file = r"D:\DeepL\data\paper\JS\ablation_study\model_F1\model_pt\50.pt"
    # args.save_path = r"D:\DeepL\data\paper\JS\ablation_study\model_F1\pre_mask1"
    # args.model_type = 'field'
    # args.test_path = r"D:\DeepL\data\paper\JS\ablation_study\model_F1\train_image"


    test_path = os.path.join(args.test_path, "*.tif")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    test_file_names = glob.glob(test_path)
    # print(test_file_names)
    # valLoader = DataLoader(DatasetImageMaskContourDist(test_file_names))
    test_file_names = [filePath.split('.')[0] for filePath in test_file_names]
    valLoader = DataLoader(DatasetImageMaskContourDist(args.test_path, test_file_names))
    # valLoader = DataLoader(DatasetImageMaskContourDist(test_file_names))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = nn.DataParallel(model)  # 自己加的
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        outputs1, outputs2 ,outputs3= model(inputs)

        ## TTA
        # outputs4, outputs5, outputs6 = model(torch.flip(inputs, [-1]))
        # predict_2 = torch.flip(outputs4, [-1])
        # outputs7, outputs8, outputs9 = model(torch.flip(inputs, [-2]))
        # predict_3 = torch.flip(outputs7, [-2])
        # outputs10, outputs11, outputs12 = model(torch.flip(inputs, [-1, -2]))
        # predict_4 = torch.flip(outputs10, [-1, -2])
        # predict_list = outputs1 + predict_2 + predict_3 + predict_4
        # pred1 = predict_list/4.0

        outputs1 = outputs1.detach().cpu().numpy().squeeze()



        res = np.zeros((256, 256))
        res[outputs1>0.5] = 255
        res[outputs1<=0.5] = 0

        res = np.array(res, dtype='uint8')
        output_path = os.path.join(
            # save_path, os.path.basename(img_file_name[0])
            save_path, os.path.basename(img_file_name[0] + ".tif")
        )
        cv2.imwrite(output_path, res)