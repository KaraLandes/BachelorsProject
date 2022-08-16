from pathlib import Path
import os
from Repo.detection.faster_rcnn.train_evaluate import TrainFRCNN
from Repo.detection.faster_rcnn.dataset import FRCNNRealBillSet
from Repo.detection.faster_rcnn.network import FastRCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

repo = Path(os.getcwd()).parent

im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")

save_dir = os.path.join(repo, "processed_data", "faster_processed", "realbills")
save_dir_unseen = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

net = FastRCNN()
net.load_state_dict(torch.load(os.path.join(repo, "progress_tracking", "detection/faster_rcnn", 'models',
                                            "run_80", 'faster_rcnn__on_ep21_new_best_model_23.0.pt')))

bills = FRCNNRealBillSet(image_dir=im_dir_real, output_shape=(80, 80), coefficient=1)
bills_big = FRCNNRealBillSet(image_dir=im_dir_real, output_shape=(4000, 4000), coefficient=1)

train_class = TrainFRCNN("", "", "", network=net)
train_class.set_device()
net = net.to(train_class.device)

loader = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)
loader_big = DataLoader(dataset=bills, batch_size=1, num_workers=1, collate_fn=train_class.collate_fn, shuffle=False)

errors = {}
errors["xmin"] = [[],[]]
errors["xmax"] = [[],[]]
errors["ymin"] = [[],[]]
errors["ymax"] = [[],[]]

for i, b in tqdm(enumerate(loader)):
    im, trg = b
    im_big, trg_big = bills_big.__getitem__(i)

    im = [el.to(train_class.device) for el in im]
    trg = [{k: v.to(train_class.device) for k, v in t.items()} for t in trg]

    pred, _ = train_class.net((im, trg, False))
    pred = pred[0]['boxes'][0].detach().cpu().numpy()
    pred_big = ((pred/im[0].shape[1])*im_big.shape[1]).astype(int)

    pred_big[1], pred_big[2] = pred_big[2], pred_big[1]

    pred_big_w = abs(pred_big[0] - pred_big[1])
    pred_big_h = abs(pred_big[2] - pred_big[3])
    r_tol = .15
    w_tolerance = int(pred_big_w * r_tol)
    h_tolerance = int(pred_big_h * r_tol)
    pred_big[0] = 0 if pred_big[0] - w_tolerance < 0 else pred_big[0] - w_tolerance
    pred_big[2] = 0 if pred_big[2] - h_tolerance < 0 else pred_big[2] - h_tolerance
    pred_big[1] = im_big.shape[-2] if pred_big[1] + w_tolerance > im_big.shape[-2] else pred_big[1] + w_tolerance
    pred_big[3] = im_big.shape[-2] if pred_big[3] + h_tolerance > im_big.shape[-2] else pred_big[3] + h_tolerance


    cropped_image = im_big[:, pred_big[0]:pred_big[1], pred_big[2]:pred_big[3]]
    cropped_mask = trg_big["masks"][:, pred_big[0]:pred_big[1], pred_big[2]:pred_big[3]]
    cropped_mask = cropped_mask.detach().cpu().numpy()

    temp_i = np.zeros((cropped_image.shape[-2], cropped_image.shape[-1], 3))
    temp_i[:, :, 0] = cropped_image[0]
    temp_i[:, :, 1] = cropped_image[1]
    temp_i[:, :, 2] = cropped_image[2]
    temp_i = temp_i.astype(np.uint8)
    Image.fromarray(temp_i).save(os.path.join(save_dir, f"real_image_{i}.jpg"))
    Image.fromarray(cropped_mask[0].astype(np.uint8)).save(os.path.join(save_dir, f"mask_real_image_{i}.jpg"))


    #errors tracking
    pred[1], pred[2] = pred[2], pred[1]
    pred_w = abs(pred[1]-pred[0])
    pred_h = abs(pred[3]-pred[2])
    x_min_diff = abs(pred[0] - trg[0]['boxes'][0][0])
    y_min_diff = abs(pred[1] - trg[0]['boxes'][0][1])
    x_max_diff = abs(pred[2] - trg[0]['boxes'][0][2])
    y_max_diff = abs(pred[3] - trg[0]['boxes'][0][3])

    errors["xmin"][0].append(x_min_diff)
    errors["xmin"][1].append(x_min_diff / pred_w)
    errors["xmax"][0].append(x_max_diff)
    errors["xmax"][1].append(x_max_diff / pred_w)
    errors["ymin"][0].append(y_min_diff)
    errors["ymin"][1].append(y_min_diff / pred_h)
    errors["ymax"][0].append(y_max_diff)
    errors["ymax"][1].append(y_max_diff / pred_h)


for k, v in errors.items():
    print(k)
    print(max(v[1]))


