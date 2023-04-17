#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import os
import time
import oss2 as oss
import torch.nn as nn

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu

from utils.meters import TrainMeter, ValMeter

from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

from datasets.utils.mixup import Mixup

logger = logging.get_logger(__name__)



@torch.no_grad()
def test_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()
    top1_per_class = {}
    num_per_class = {}

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        if misc.get_num_gpus(cfg):

            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if misc.get_num_gpus(cfg) > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'PCMGenerator'):
            preds, logits = model(inputs)
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        else:
            
            model_dict = model(task_dict)
            target_logits = model_dict['logits']
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE

            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                labels = task_dict['target_labels']
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                _top_max_k_vals, top_max_k_inds = torch.topk(
                                                        preds, 1, dim=1, largest=True, sorted=True
                                                    )
                for index, score in enumerate(top_max_k_inds):
                    if str(task_dict['real_target_labels'][index].cpu().item()) in num_per_class:
                        num_per_class[str(task_dict['real_target_labels'][index].cpu().item())] += 1
                        # top1_per_class[str(task_dict['real_target_labels'].cpu().item())] += 1
                    else:
                        num_per_class[str(task_dict['real_target_labels'][index].cpu().item())] = 1
                    if str(task_dict['real_target_labels'][index].cpu().item()) not in top1_per_class:
                        top1_per_class[str(task_dict['real_target_labels'][index].cpu().item())] = 0
                    if score[0] == labels[index]:
                        top1_per_class[str(task_dict['real_target_labels'][index].cpu().item())] += 1


                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                val_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=len(val_loader) * cur_epoch + cur_iter,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    for perclass in top1_per_class:
        # top1_per_class[perclass] = num_per_class[perclass]
        logger.info("class: {}, acc: {}".format(perclass, top1_per_class[perclass]/num_per_class[perclass]))
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if misc.get_num_gpus(cfg):
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()

def test_few_shot(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("TEST with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create the video train and val loaders.
    val_loader = build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None  # val

    # Create meters.
    if cfg.DETECTION.ENABLE:
        # train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        # train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        misc.get_num_gpus(cfg)
    ):
        # writer = tb.TensorboardWriter(cfg)
        pass
    else:
        writer = None

    cur_epoch = 0
    test_epoch(
        val_loader, model, val_meter, cur_epoch,  cfg, writer
    )

    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

