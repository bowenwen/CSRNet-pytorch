import torch as torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm

from config import Config
from model import CSRNet
from dataset import create_train_dataloader, create_test_dataloader, create_test_extra_dataloader
from utils import denormalize


def cal_mae(img_root, gt_dmap_root, model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    model = CSRNet()
    model.load_state_dict(torch.load(
        model_param_path, map_location=cfg.device))
    model.to(cfg.device)
    test_dataloader = create_test_dataloader(
        cfg.dataset_root)             # dataloader
    model.eval()
    sum_mae = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            # forward propagation
            et_densitymap = model(image).detach()
            mae = abs(et_densitymap.data.sum()-gt_densitymap.data.sum())
            sum_mae += mae.item()
            # clear mem
            del i, data, image, gt_densitymap, et_densitymap
            torch.cuda.empty_cache()

    print("model_param_path:"+model_param_path +
          " mae:"+str(sum_mae/len(test_dataloader)))


def estimate_density_map(img_root, gt_dmap_root, model_param_path, index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    image_export_folder = 'export_images'
    model = CSRNet()
    model.load_state_dict(torch.load(
        model_param_path, map_location=cfg.device))
    model.to(cfg.device)
    test_dataloader = create_test_dataloader(
        cfg.dataset_root)             # dataloader
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            image = data['image'].to(cfg.device)
            gt_densitymap = data['densitymap'].to(cfg.device)
            # forward propagation
            et_densitymap = model(image).detach()
            pred_count = et_densitymap.data.sum().cpu()
            actual_count = gt_densitymap.data.sum().cpu()
            et_densitymap = et_densitymap.squeeze(0).squeeze(0).cpu().numpy()
            gt_densitymap = gt_densitymap.squeeze(0).squeeze(0).cpu().numpy()
            image = image[0].cpu()  # denormalize(image[0].cpu())
            print(et_densitymap.shape)
            # et is the estimated density
            plt.imshow(et_densitymap, cmap=CM.jet)
            plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
                                                str(i).zfill(3),
                                                str(int(pred_count)),
                                                str(int(actual_count)), 'etdm.png'))
            # gt is the ground truth density
            plt.imshow(gt_densitymap, cmap=CM.jet)
            plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
                                                str(i).zfill(3),
                                                str(int(pred_count)),
                                                str(int(actual_count)), 'gtdm.png'))
            # image
            plt.imshow(image.permute(1, 2, 0))
            plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
                                                str(i).zfill(3),
                                                str(int(pred_count)),
                                                str(int(actual_count)), 'image.png'))

            # clear mem
            del i, data, image, et_densitymap, gt_densitymap, pred_count, actual_count
            torch.cuda.empty_cache()


def estimate_density_map_no_gt(img_root, gt_dmap_root, model_param_path, index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    image_export_folder = 'export_images_extra'
    model = CSRNet()
    model.load_state_dict(torch.load(
        model_param_path, map_location=cfg.device))
    model.to(cfg.device)
    test_dataloader = create_test_extra_dataloader(
        cfg.dataset_root)             # dataloader
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            image = data['image'].to(cfg.device)
            # gt_densitymap = data['densitymap'].to(cfg.device)
            # forward propagation
            et_densitymap = model(image).detach()
            pred_count = et_densitymap.data.sum().cpu()
            # actual_count = gt_densitymap.data.sum().cpu()
            actual_count = 999
            et_densitymap = et_densitymap.squeeze(0).squeeze(0).cpu().numpy()
            # gt_densitymap = gt_densitymap.squeeze(0).squeeze(0).cpu().numpy()
            image = image[0].cpu()  # denormalize(image[0].cpu())
            print(et_densitymap.shape)
            # et is the estimated density
            plt.imshow(et_densitymap, cmap=CM.jet)
            plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
                                                str(i).zfill(3),
                                                str(int(pred_count)),
                                                str(int(actual_count)), 'etdm.png'))
            # # gt is the ground truth density
            # plt.imshow(gt_densitymap, cmap=CM.jet)
            # plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
            #                                     str(i).zfill(3),
            #                                     str(int(pred_count)),
            #                                     str(int(actual_count)), 'gtdm.png'))
            # image
            plt.imshow(image.permute(1, 2, 0))
            plt.savefig("{}/{}_{}_{}_{}".format(image_export_folder,
                                                str(i).zfill(3),
                                                str(int(pred_count)),
                                                str(int(actual_count)), 'image.png'))

            # clear mem
            del i, data, image, et_densitymap, pred_count, actual_count
            torch.cuda.empty_cache()


if __name__ == "__main__":
    cfg = Config()
    torch.backends.cudnn.enabled = False
    img_root = cfg.dataset_root
    gt_dmap_root = 'test'
    model_param_path = './checkpoints/69.pth'
    cal_mae(img_root, gt_dmap_root, model_param_path)
    estimate_density_map(img_root, gt_dmap_root, model_param_path, index=None)

    # a little bit of a hack to run model on a set of images with no ground truth
    gt_dmap_root = 'test_extra'
    estimate_density_map_no_gt(
        img_root, gt_dmap_root, model_param_path, index=None)

    # results
    # model_param_path:./checkpoints/95.pth mae:68.33428703559623
