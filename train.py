import glob
import logging
import os
import random
import torch
# from dataset import DatasetImageMaskContourDist
from dataset_RE import DatasetImageMaskContourDist
from losses import LossF
from models import BsiNet,Field
# from model_noEdgeBrunch import Field
# from model_Lap import Field
# from models_Baseline import Field
# from models_Baseline_GLCA import Field
# from models_Baseline_GLCA_BGM import Field
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import visualize, create_train_arg_parser,evaluate
# from torchsummary import summary
from sklearn.model_selection import train_test_split

###可以试试调整边界的权重研究影响，但是可能会麻烦点，后面再看吧
def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "field":
        criterion = LossF(weights)

    return criterion


def build_model(model_type):

    if model_type == "field":
        model = Field(num_classes=2)

    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "field":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
            loss.backward()
            optimizer.step()

    return loss


if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()
    # args.pretrained_model_path = r"E:\zhaohang\data\onesoil_2m_pt\56.pt"
    args.pretrained_model_path = r"E:\zhaohang\DeepL\模型训练pt文件_lmx机子\dikuai_all_ihave_ZH\60.pt"
    # args.pretrained_model_path = r"F:\PingAn\yangben_Round2\model_pt\90.pt"
    # args.pretrained_model_path = r"F:\PingAn\NMG\pt_file\40.pt"

    # args.train_path = r"D:\DeepL\data\paper\HB\train_image"
    # args.model_type = 'field'
    # args.save_path = r'D:\DeepL\data\paper\HB\model_zh_2renwu'
    # args.train_path = r"D:\DeepL\data\paper\JS\train_image"
    # args.model_type = 'field'
    # args.save_path = r"D:\DeepL\data\paper\JS\ablation_study\model_baseline\model_pt"
    args.train_path = r"F:\PingAn\NX\Ningxia\output\train_image"
    args.model_type = 'field'
    args.save_path = r"F:\PingAn\NX\Ningxia\output\model_pt"

    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")

    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"))
    random.shuffle(train_file_names)

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.01, random_state=41)  # 0.2太多了  0.01都行

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    print(device)
    model = build_model(args.model_type)

    if torch.cuda.device_count() > 0:           #本来是0
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
  #  summary(model, input_size=(3, 256, 256))

    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)
    print('train',args.use_pretrained)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,train_file),
        batch_size=args.batch_size,drop_last=False,  shuffle=True
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file),drop_last=False,
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file),
        batch_size=args.val_batch_size,drop_last=False, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  #  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(1e10), eta_min=1e-5)
   # scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1)    #新加的
    criterion = define_loss(args.model_type)


    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            targets = [targets1, targets2, targets3]


            loss = train_model(model, targets, args.model_type, criterion, optimizer)

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        epoch_loss = running_loss / len(train_file_names)
        print(epoch_loss)

        if epoch % 1 == 0:

            dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar("loss_valid", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss, dev_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))


        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )


