import os

import torch
from torchsummary import summary

from pathlib import Path
from torch.optim import Adam

# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
DUMMY = False  # detect bills on images
FASTRCNN = False
JOINT_CORNERS_DETECTOR = True
CORNERS_REFINEMENT = False

repo = Path(os.getcwd())
# repo = repo.parent.absolute()

########################################################################################################################
# Stage 3
# In this stage I use generated bills to train a CNN which returns cleared bill image only
# This is dummy network
########################################################################################################################
im_dir_gen = os.path.join(repo, "processed_data", "genbills")
im_dir_real = os.path.join(repo, "processed_data", "realbills")
im_dir_unseen = os.path.join(repo, "processed_data", "realbills", "unseen")
if DUMMY:
    from detection.dummy_cnn.dataset import BaseBillOnBackGroundSet, BaseRealBillSet
    from detection.dummy_cnn.network import BaseNetDetect
    from detection.dummy_cnn.train_evaluate import TrainBaseDetect

    # Baseline Network (CNN)
    network = BaseNetDetect(n_hidden_layers=8, kernel_size=2)

    train_class = TrainBaseDetect(im_dir_gen=im_dir_gen,
                                  im_dir_real=im_dir_real,
                                  im_dir_unseen=im_dir_unseen,
                                  network=network, )
    train_class.set_datasets(train_dataset_type=BaseBillOnBackGroundSet,
                             valid_dataset_type=BaseRealBillSet,
                             output_shape=(128, 128), coefficient=1)
    train_class.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "detection/dummy_cnn", "tensorboard"))
    train_class.set_loaders(batch_size=1)
    train_class.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "detection/dummy_cnn", "models", "basenet_")
    save_images_path = os.path.join(repo, "progress_tracking", "detection/dummy_cnn", "visualization")

    train_class.train(optimizer=Adam(network.parameters(), lr=8e-4, weight_decay=1e-5),
                      save_model_path=save_model_path, epochs=100, method=train_class.depict,
                      save_images_path=save_images_path)
if FASTRCNN:
    from detection.faster_rcnn.dataset import FRCNNBillOnBackGroundSet, FRCNNRealBillSet
    from detection.faster_rcnn.network import FastRCNN
    from detection.faster_rcnn.train_evaluate import TrainFRCNN

    network = FastRCNN()
    for p in network.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    train_class = TrainFRCNN(im_dir_gen=im_dir_gen,
                             im_dir_real=im_dir_real,
                             im_dir_unseen=im_dir_unseen,
                             network=network)
    train_class.set_datasets(train_dataset_type=FRCNNBillOnBackGroundSet,
                             valid_dataset_type=FRCNNRealBillSet, coefficient=1,
                             output_shape=(128, 128))
    train_class.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "tensorboard"))
    train_class.set_loaders(batch_size=2)
    train_class.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "models", "faster_rcnn_")
    save_images_path = os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "visualization")

    train_class.train(optimizer=Adam(network.parameters(), lr=5e-4, weight_decay=5e-5),
                      save_model_path=save_model_path, epochs=100, method=train_class.depict,
                      save_images_path=save_images_path)
if JOINT_CORNERS_DETECTOR:
    from detection.corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet
    from detection.corners_nn.network import CornerDetector
    from detection.corners_nn.train_evaluate import TrainCorner

    network = CornerDetector(compute_attention=True)
    for p in network.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    train_class = TrainCorner(im_dir_gen=im_dir_gen,
                              im_dir_real=im_dir_real,
                              im_dir_unseen=im_dir_unseen,
                              network=network)
    train_class.set_datasets(train_dataset_type=CornerBillOnBackGroundSet,
                             valid_dataset_type=CornerRealBillSet, coefficient=1,
                             output_shape=(128, 128))
    train_class.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "detection/corners_nn", "tensorboard"))
    train_class.set_loaders(batch_size=8)
    train_class.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "detection/corners_nn", "models", "corners_nn")
    save_images_path = os.path.join(repo, "progress_tracking", "detection/corners_nn", "visualization")

    network.vgg16_enc.requires_grad_(False)
    params = [p for p in network.parameters() if p.requires_grad == True]
    opt = Adam(params, lr=3e-4, weight_decay=5e-5)

    train_class.train(optimizer=opt,
                      save_model_path=save_model_path, epochs=81, method=train_class.depict_corners,
                      save_images_path=save_images_path)

if CORNERS_REFINEMENT:
    from detection.refine_nn.dataset import RefineBillOnBackGroundSet, RefineRealBillSet
    from detection.refine_nn.network import RefineNet
    from detection.refine_nn.train_evaluate import TrainRefine

    # we have 4 patterns and therefore 4 training processes for red, blue, green, yellow
    for type in ['red', 'green', 'blue', 'yellow']:
        network = RefineNet(net_type=type)
        for p in network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        train_class = TrainRefine(im_dir_gen=im_dir_gen,
                                  im_dir_real=im_dir_real,
                                  im_dir_unseen=im_dir_unseen,
                                  network=network)
        train_class.set_datasets(train_dataset_type=RefineBillOnBackGroundSet,
                                 valid_dataset_type=RefineRealBillSet, coefficient=1,
                                 output_shape=(128, 128))
        train_class.set_writer(
            log_dir=os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "tensorboard"))
        train_class.set_loaders(batch_size=8, workers=12)
        train_class.set_device()
        save_model_path = os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "models", "corners_nn")
        save_images_path = os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "visualization")

        network.vgg.requires_grad_(False)
        params = [p for p in network.parameters() if p.requires_grad == True]
        opt = Adam(params, lr=1e-3, weight_decay=5e-5)
        train_class.train(optimizer=opt,
                          save_model_path=save_model_path, epochs=30, method=train_class.depict_refinement,
                          save_images_path=save_images_path)
