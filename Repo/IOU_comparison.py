import operator
from pathlib import Path
import os
from Repo.detection.faster_rcnn.train_evaluate import TrainFRCNN
from Repo.detection.faster_rcnn.dataset import FRCNNRealBillSet, FRCNNBillOnBackGroundSet
from Repo.detection.dummy_cnn.dataset import BaseRealBillSet, BaseBillOnBackGroundSet
from Repo.detection.faster_rcnn.network import FasterRCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

repo = Path(os.getcwd())

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")

im_dir_real_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills")
im_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

save_dir = os.path.join(repo, "processed_data", "faster_processed_2", "realbills")
save_dir_unseen = os.path.join(repo, "processed_data", "faster_processed_2", "realbills", "unseen")

UNSEEN = 2
if UNSEEN==1:
    im_dir = im_dir_unseen_faster
    save_dir = save_dir_unseen
elif UNSEEN==0:
    im_dir = im_dir_real_faster
    save_dir = save_dir

elif UNSEEN==2:
    im_dir = im_dir_gen
    save_dir = save_dir
# # FRCNN##################################################################################################################
#
# net = FasterRCNN()
# net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/faster_rcnn", 'models',
#                                             "64",
#                                             "faster_rcnn__on_ep8_new_best_model_25.0.pt"),
#                                map_location=torch.device('cpu')))
#
# if UNSEEN!=2:
#     bills = FRCNNRealBillSet(image_dir=im_dir, output_shape=(64, 64), coefficient=1)
#     bills_big = BaseRealBillSet(image_dir=im_dir, output_shape=(1000, 1000), coefficient=1)
# else:
#     bills = FRCNNBillOnBackGroundSet(image_dir=im_dir_gen, output_shape=(64, 64), coefficient=1)
#     bills_big = FRCNNBillOnBackGroundSet(image_dir=im_dir_gen, output_shape=(1000, 1000), coefficient=1)
#
# train_class = TrainFRCNN("", "", "", network=net)
# train_class.set_device(cpu=True)
# net = net.to(train_class.device)
#
# loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=True)
# loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=True)
#
# ious_b = []
# ious_m = []
#
# for i, b in tqdm(enumerate(loader)):
#     if i == 200:
#         break
#     im, trg = b
#     im_big, trg_big = bills_big.__getitem__(i)
#
#     im = [el.to(train_class.device) for el in im]
#     trg = [{k: v.to(train_class.device) for k, v in t.items()} for t in trg]
#
#     pred, _ = train_class.net((im, trg, False))
#     pred = pred[0]['boxes'][0].detach().cpu().numpy()
#     pred_big = ((pred/im[0].shape[1])*im_big.shape[1]).astype(int)
#
#     # compute IOU of bboxes####################################################################
#     # intersection
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(pred_big[0], trg_big["boxes"][0][0])
#     yA = max(pred_big[1], trg_big["boxes"][0][1])
#     xB = min(pred_big[2], trg_big["boxes"][0][2])
#     yB = min(pred_big[3], trg_big["boxes"][0][3])
#
#     # compute the area of intersection rectangle
#     interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
#     if interArea == 0:
#         ious_b.append(0)
#         continue
#
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = abs((pred_big[2] - pred_big[0]) * (pred_big[3] - pred_big[1]))
#     boxBArea = abs((trg_big["boxes"][0][2] - trg_big["boxes"][0][0]) * (trg_big["boxes"][0][3] - trg_big["boxes"][0][1]))
#
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the intersection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     ious_b.append(iou)
#
#     # compute IOU of predicted bbox and mask####################################################################
#     # intersection of mask and bbox
#
#     # create a mask from bbox
#     bbox_mask = np.zeros((1000, 1000))
#     bbox_mask[pred_big[0]:pred_big[2], pred_big[1]:pred_big[3]] = 1
#
#     #sum 2 masks
#     # inverted_target_mask = np.zeros((1000, 1000))
#     # inverted_target_mask[trg_big["masks"][0]==0]=1 #gen vs real different masks
#     summed_mask = bbox_mask+trg_big["masks"][0].numpy()
#     iou = np.where(summed_mask>1)[0].shape[0]/np.where(summed_mask>0)[0].shape[0]
#     ious_m.append(iou)
#
# print("Test data") if UNSEEN else print("Train data")
# print("bboxes iou", np.mean(ious_b))
# print("bbox vs mask iou", np.mean(ious_m))

# JCD##################################################################################################################
from Repo.detection.corners_nn.network import CornerDetector
from Repo.detection.corners_nn.train_evaluate import TrainCorner
from Repo.detection.corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet
import rasterio.features
from shapely.geometry.polygon import Polygon

SHAPE = 256
net = CornerDetector(compute_attention=True, size=SHAPE).to('cpu')
net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn_c_jcd", 'models',
                                            str(SHAPE),
                                            'corners_nn_on_ep10_new_best_model_159.0.pt'),
                               map_location=torch.device('cpu')))

if UNSEEN != 2:
    bills = CornerRealBillSet(image_dir=im_dir, output_shape=(SHAPE, SHAPE), coefficient=1)
    bills_big = CornerRealBillSet(image_dir=im_dir, output_shape=(1000, 1000), coefficient=1)
else:
    bills = CornerBillOnBackGroundSet(image_dir=im_dir, output_shape=(SHAPE, SHAPE), coefficient=1)
    bills_big = CornerBillOnBackGroundSet(image_dir=im_dir, output_shape=(1000, 1000), coefficient=1)

train_class = TrainCorner("", "", "", network=net)
train_class.set_device(cpu=True)

loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)
loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)

ious = []
for i, b in tqdm(enumerate(loader), total=1500):
    if i%2 == 0:
        continue
    if i%3 == 0:
        continue
    if i%5 == 0:
        continue
    if i%7 == 0:
        continue
    if i%11 == 0:
        continue
    if i%13 == 0:
        continue
    if len(ious)==200: break
    im, corner_trg, mask_trg = b
    im_big, trg_big = bills_big.__getitem__(i)
    corner_trg_big, mask_trg_big = trg_big

    im = im.to('cpu')
    corner_trg = corner_trg.to('cpu')
    pred, _ = net(im)
    prediction = pred[0].to("cpu").detach().numpy()
    prediction = [el if el > 0 else 0 for el in prediction]
    prediction = np.array([el if el < SHAPE-1 else SHAPE-1 for el in prediction])
    prediction_big = prediction * im_big.shape[-1] / im.shape[-1]
    prediction_big = prediction_big.reshape(4, 2)

    polygon_pred = Polygon(prediction_big)
    polygon_pred = rasterio.features.rasterize([polygon_pred], out_shape=(1000, 1000))

    polygon_trg = Polygon(corner_trg_big)
    polygon_trg = rasterio.features.rasterize([polygon_trg], out_shape=(1000, 1000))

    #sum 2 masks
    summed_mask = polygon_pred+polygon_trg
    iou = np.where(summed_mask>1)[0].shape[0]/np.where(summed_mask>0)[0].shape[0]
    ious.append(iou)

print("Test data") if UNSEEN else print("Train data")
print("bboxes iou", np.mean(ious))

# #CR##################################################################################################################
# from Repo.detection.refine_nn.train_evaluate import TrainRefine
# from Repo.detection.refine_nn.dataset import RefineRealBillSet
# from Repo.detection.refine_nn.network import RefineNet
# import rasterio.features
# from shapely.geometry.polygon import Polygon
# from Repo.detection.corners_nn.network import CornerDetector
#
# SHAPE = 256
# jcd = CornerDetector(compute_attention=True, size=SHAPE).to('cpu')
# jcd.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn_c_jcd", 'models',
#                                             str(SHAPE),
#                                             'corners_nn_on_ep10_new_best_model_159.0.pt'),
#                     map_location=torch.device('cpu')))
#
#
# crn = RefineNet(net_type="red")#dummy instance
# bills = RefineRealBillSet(image_dir=im_dir, output_shape=(SHAPE, SHAPE), coefficient=1)
# train_class = TrainRefine("", "", "", network=crn)
# train_class.set_device(cpu=True)
# loader = DataLoader(dataset=bills, batch_size=1, num_workers=11, collate_fn=train_class.collate_fn,
#                     shuffle=False)
# best_nets = [os.path.join(repo, "progress_tracking", "detection/refine_nn_c_cr", "red", 'models',
#                           'models_on_ep15_new_best_model_60.0.pt'),
#              os.path.join(repo, "progress_tracking", "detection/refine_nn_c_cr", "green", 'models',
#                           'models_on_ep20_new_best_model_74.0.pt'),
#              os.path.join(repo, "progress_tracking", "detection/refine_nn_c_cr", "blue", 'models',
#                           'models_on_ep13_new_best_model_85.0.pt'),
#              os.path.join(repo, "progress_tracking", "detection/refine_nn_c_cr", "yellow", 'models',
#                           'models_on_ep20_new_best_model_64.0.pt')
#              ]
# colors = ["red", 'green', 'blue', 'yellow']
# ious = []
# for im, _, _, originals, corners_big in tqdm(loader):
#     coords_pred, _ = jcd(im.to("cpu"))
#
#     # recalculation of predicted corners
#     coords_pred = torch.reshape(coords_pred, shape=(len(coords_pred), 4, 2))
#     corners_big = torch.reshape(corners_big, shape=(len(corners_big), 4, 2))
#
#     big_corners_pred = torch.stack([torch.stack([train_class.rescale_corner(c, im.shape[-2:], originals.shape[-2:])
#                                                  for c in el]) for el in coords_pred])
#
#     refine_corners_pred = []
#     for active_id, type, state_dict in zip(range(len(colors)), colors, best_nets):
#         crn = RefineNet(net_type=type)
#         state_dict = torch.load(state_dict, map_location=train_class.device)
#         crn.load_state_dict(state_dict)
#         train_class.net = crn.to(train_class.device)
#
#         p, _, changes = train_class.crop(original=train_class.numpy(originals[0]),
#                                          old_prediction=train_class.numpy(big_corners_pred[0][active_id]),
#                                          changes={},
#                                          mask=np.zeros(originals[0].shape[-2:]))
#         sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=False)
#         p = torch.tensor(p).to(train_class.device)
#
#         o_p_big = big_corners_pred[0][active_id]
#         o_p_for_net = train_class.apply_changes(o_p_big, sorted_changes)
#
#         o_p_for_net = o_p_for_net / p.shape[-1]  # relative
#         n_p, mask_pred = train_class.net((p, o_p_for_net))
#         n_p_on_patch = n_p * p.shape[-2]
#         sorted_changes = sorted(changes.items(), key=operator.itemgetter(0), reverse=True)
#         n_p_big = train_class.untrack_changes(n_p_on_patch, sorted_changes)
#
#         pred_point, _ = train_class.refine_loop(n_p_big=n_p_big,
#                                                 o_p_big=o_p_big,
#                                                 target=corners_big[0][active_id],
#                                                 changes=changes,
#                                                 patch=p)
#         refine_corners_pred.append(train_class.numpy(pred_point))
#
#     polygon_pred = Polygon(refine_corners_pred)
#     polygon_pred = rasterio.features.rasterize([polygon_pred], out_shape=(1000, 1000))
#
#     polygon_trg = Polygon(corners_big[0].detach().numpy())
#     polygon_trg = rasterio.features.rasterize([polygon_trg], out_shape=(1000, 1000))
#
#     #sum 2 masks
#     summed_mask = polygon_pred+polygon_trg
#     iou = np.where(summed_mask>1)[0].shape[0]/np.where(summed_mask>0)[0].shape[0]
#     ious.append(iou)
#
# print("Test data") if UNSEEN else print("Train data")
# print("bboxes iou", np.mean(ious))
