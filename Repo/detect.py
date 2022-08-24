import os

import torch
from torchsummary import summary

from pathlib import Path
from torch.optim import Adam

# This file is created to prepare and preprocess all  data
# HYPERPARAMETERS, WHICH CODE TO RUN
DUMMY = 0  # detect bills on images
FASTRCNN = 0
JOINT_CORNERS_DETECTOR = 0
CORNERS_REFINEMENT = 1

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

im_dir_real_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills")
im_dir_unseen_faster = os.path.join(repo, "processed_data", "faster_processed", "realbills", "unseen")

im_dir_real_faster2 = os.path.join(repo, "processed_data", "faster_processed_2", "realbills")
im_dir_unseen_faster2 = os.path.join(repo, "processed_data", "faster_processed_2", "realbills", "unseen")

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
    state_dict = torch.load(os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "models",
                                         "run_48_crop", "faster_rcnn__on_ep10_new_best_model_6.0.pt"))
    network.load_state_dict(state_dict)
    for p in network.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    train_class = TrainFRCNN(im_dir_gen=im_dir_gen,
                             im_dir_real=im_dir_real_faster,
                             im_dir_unseen=im_dir_unseen_faster,
                             network=network)
    train_class.set_datasets(train_dataset_type=FRCNNBillOnBackGroundSet,
                             valid_dataset_type=FRCNNRealBillSet, coefficient=1,
                             output_shape=(48, 48))
    train_class.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "tensorboard"))
    train_class.set_loaders(batch_size=2)
    train_class.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "models", "faster_rcnn_")
    save_images_path = os.path.join(repo, "progress_tracking", "detection/faster_rcnn", "visualization")

    train_class.train(optimizer=Adam(network.parameters(), lr=1e-7, weight_decay=5e-5),
                      save_model_path=save_model_path, epochs=20, method=train_class.depict,
                      save_images_path=save_images_path)

if JOINT_CORNERS_DETECTOR:
    from detection.corners_nn.dataset import CornerRealBillSet, CornerBillOnBackGroundSet
    from detection.corners_nn.network import CornerDetector
    from detection.corners_nn.train_evaluate import TrainCorner

    network = CornerDetector(compute_attention=True)
    state_dict = torch.load(os.path.join(repo, "progress_tracking", "detection/corners_nn", "models",
                                         "run_64_crop", "retrain_2",
                                         "corners_nn_on_ep3_new_best_model_22.0.pt"))

    network.load_state_dict(state_dict)
    for p in network.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    train_class = TrainCorner(im_dir_gen=im_dir_gen,
                              im_dir_real=im_dir_real_faster2,
                              im_dir_unseen=im_dir_unseen_faster2,
                              network=network)
    train_class.set_datasets(train_dataset_type=CornerBillOnBackGroundSet,
                             valid_dataset_type=CornerRealBillSet, coefficient=1,
                             output_shape=(64, 64))
    train_class.set_writer(
        log_dir=os.path.join(repo, "progress_tracking", "detection/corners_nn", "tensorboard"))
    train_class.set_loaders(batch_size=8, workers=11)
    train_class.set_device()
    save_model_path = os.path.join(repo, "progress_tracking", "detection/corners_nn", "models", "corners_nn")
    save_images_path = os.path.join(repo, "progress_tracking", "detection/corners_nn", "visualization")

    network.vgg16_enc.requires_grad_(False)
    params = [p for p in network.parameters() if p.requires_grad == True]
    opt = Adam(params, lr=8e-4, weight_decay=5e-5)

    train_class.train(optimizer=opt,
                      save_model_path=save_model_path, epochs=30, method=train_class.depict_corners,
                      save_images_path=save_images_path)

if CORNERS_REFINEMENT:
    from detection.refine_nn.dataset import RefineBillOnBackGroundSet, RefineRealBillSet
    from detection.refine_nn.network import RefineNet
    from detection.refine_nn.train_evaluate import TrainRefine

    # we have 4 patterns and therefore 4 training processes for red, blue, green, yellow
    for type in ['red', 'green', 'blue', 'yellow']: # ['red', 'green', 'blue', 'yellow']
        print()
        print("="*110)
        print(type)
        print("=" * 110)
        print()
        network = RefineNet(net_type=type)
        for p in network.parameters():
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        train_class = TrainRefine(im_dir_gen=im_dir_gen,
                                  im_dir_real=im_dir_real_faster2,
                                  im_dir_unseen=im_dir_unseen_faster2,
                                  network=network)
        train_class.set_datasets(train_dataset_type=RefineBillOnBackGroundSet,
                                 valid_dataset_type=RefineRealBillSet, coefficient=1,
                                 output_shape=(48, 48))
        train_class.set_writer(
            log_dir=os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "tensorboard"))
        train_class.set_loaders(batch_size=8, workers=11)
        train_class.set_device()
        save_model_path = os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "models", "corners_nn")
        save_images_path = os.path.join(repo, "progress_tracking", "detection/refine_nn", type, "visualization")

        network.vgg.requires_grad_(False)
        params = [p for p in network.parameters() if p.requires_grad == True]
        opt = Adam(params, lr=1e-6, weight_decay=5e-6)
        train_class.train(optimizer=opt,
                          save_model_path=save_model_path, epochs=40, method=train_class.depict_refinement,
                          save_images_path=save_images_path)
