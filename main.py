import os
import random
import argparse
import yaml
from tqdm import tqdm
import time
import copy
from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from datasets.imagenet import ImageNet

from datasets import build_dataset
from datasets.utils import build_data_loader
from InjectedClip import InjectedClip
import clip
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def finetune_model(cfg, injected_clip_model, train_loader, val_loader, test_loader, dataset, branch):
    lr = cfg["lr"]
    epochs = cfg["train_epoch"]
    clip_model = injected_clip_model.clip_model
    trained_model = clip_model
    if branch == "visual":
        trained_model = trained_model.visual
    if branch == "text":
        trained_model = trained_model.transformer

        # we don't have to pass the validation data through the image encoder each time if we only train the text
        # encoder because in that case the validation features don't change
        val_features, val_labels = load_features(clip_model, val_loader)

    optimizer = torch.optim.AdamW(trained_model.parameters(), lr=lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    best_acc, best_epoch = 0.0, 0

    if branch == "visual":
        with torch.no_grad():
            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    for train_idx in range(epochs):
        trained_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, epochs))

        for _, (images, target) in enumerate(tqdm(train_loader)):
            if branch == "text":
                with torch.no_grad():
                    images, target = images.cuda(), target.cuda()
                    image_features_raw = clip_model.encode_image(images)
                    image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)
            else:
                images, target = images.cuda(), target.cuda()
                image_features_raw = clip_model.encode_image(images)
                image_features = image_features_raw / image_features_raw.norm(dim=-1, keepdim=True)

            del images
            del image_features_raw

            if branch != "visual":
                clip_weights = clip_classifier_with_grad(dataset.classnames, dataset.template, clip_model)
            clip_logits = 100. * image_features @ clip_weights
            loss = F.cross_entropy(clip_logits, target)

            acc = cls_acc(clip_logits, target)
            correct_samples += acc / 100 * len(clip_logits)
            all_samples += len(clip_logits)
            loss_list.append(loss.item())
            del clip_logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        clip_model.eval()

        # we have to pass the validation data through the image encoder each time if we train the visual encoder because
        # in that case the validation features change
        if branch != "text":
            val_features, val_labels = [], []
            with torch.no_grad():
                for _, (images, target) in enumerate(tqdm(val_loader)):
                    images, target = images.cuda(), target.cuda()
                    image_features = clip_model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    val_features.append(image_features)
                    val_labels.append(target)

            val_features, val_labels = torch.cat(val_features), torch.cat(val_labels)

        clip_logits = 100. * val_features @ clip_weights
        acc = cls_acc(clip_logits, val_labels)

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            injected_state_dict = copy.deepcopy(injected_clip_model.get_injected_state_dict())

    clip_model.eval()
    injected_clip_model_dict = injected_clip_model.state_dict()
    injected_clip_model_dict.update(injected_state_dict)
    injected_clip_model.load_state_dict(injected_clip_model_dict)

    print(f"best epoch: {best_epoch}, accuracy: {best_acc}")
    current_lr = scheduler.get_last_lr()[0]
    print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                   correct_samples, all_samples,
                                                                   sum(loss_list) / len(loss_list)))

    test_features, test_labels = load_features(clip_model, test_loader)

    clip_logits = 100. * test_features @ clip_weights
    test_acc = cls_acc(clip_logits, test_labels)
    print(f"\n**** Test accuracy for branch {branch}: {test_acc:.2f}. ****\n")

    return clip_model


def get_inject_adapter(in_ft, rank, out_ft, clip_model):
    return nn.Sequential(OrderedDict([
        ("Linear1", nn.Linear(in_ft, rank, bias=False)),
        ("ReLU1", nn.ReLU(inplace=True)),
        ("Linear2", nn.Linear(rank, out_ft, bias=False)),
        ("ReLU2", nn.ReLU(inplace=True))
    ])).to(clip_model.dtype)


def main():
    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    print("\nRunning configs.")

    random.seed(1)
    torch.manual_seed(1)

    # here we run 3 different experiments where we either inject into both the visual and the textual branch ("both"),
    # only the textual branch ("text") or only the visual branch ("visual") of the CLIP backbone
    for branch in ["both", "text", "visual"]:
        # CLIP
        clip_model, preprocess = clip.load(cfg['backbone'])

        visual_width = clip_model.visual.transformer.width
        text_width = clip_model.transformer.width

        # here, we inject a bottleneck model inspired by CLIP-Adapter consisting of 2 linear layers
        # followed by 2 ReLUs into the first linear layer in each MLP module ("c_fc")
        bottleneck_rank = 8

        inject_c_fc_visual = get_inject_adapter(visual_width, bottleneck_rank, visual_width * 4, clip_model)
        nn.init.constant_(inject_c_fc_visual.Linear1.weight, 1 / visual_width)
        nn.init.constant_(inject_c_fc_visual.Linear2.weight, 1 / (visual_width * bottleneck_rank * 4))

        inject_c_fc_text = get_inject_adapter(text_width, bottleneck_rank, text_width * 4, clip_model)
        nn.init.constant_(inject_c_fc_text.Linear1.weight, 1 / text_width)
        nn.init.constant_(inject_c_fc_text.Linear2.weight, 1 / (text_width * bottleneck_rank * 4))

        # in addition to the bottleneck layers, we inject low-rank decomposition matrices (LoRA) into the query and
        # key layers in each Multihead-Attention module
        lora_rank = 8

        inject_mha_visual = nn.Sequential(OrderedDict([
            ("Dropout", nn.Dropout(p=0.01)),
            ("A", nn.Linear(visual_width, lora_rank, bias=False)),
            ("B", nn.Linear(lora_rank, visual_width, bias=False)),
        ])).to(clip_model.dtype)
        nn.init.kaiming_uniform_(inject_mha_visual.A.weight, a=math.sqrt(5))
        nn.init.zeros_(inject_mha_visual.B.weight)

        inject_mha_text = nn.Sequential(OrderedDict([
            ("Dropout", nn.Dropout(p=0.01)),
            ("A", nn.Linear(text_width, lora_rank, bias=False)),
            ("B", nn.Linear(lora_rank, text_width, bias=False)),
        ])).to(clip_model.dtype)

        nn.init.kaiming_uniform_(inject_mha_text.A.weight, a=math.sqrt(5))
        nn.init.zeros_(inject_mha_text.B.weight)

        if branch == "both":
            injected_layers = {"visual.mlp": {"c_fc": inject_c_fc_visual},
                               "text.mlp": {"c_fc": inject_c_fc_text},
                               "visual.mha": {"query": inject_mha_visual, "key": inject_mha_visual},
                               "text.mha": {"query": inject_mha_text, "key": inject_mha_text}
                               }

        else:
            if branch == "text":
                inject_c_fc = inject_c_fc_text
                inject_mha = inject_mha_text
            else:
                inject_c_fc = inject_c_fc_visual
                inject_mha = inject_mha_visual

            injected_layers = {f"{branch}.mlp": {"c_fc": inject_c_fc},
                               f"{branch}.mha": {"query": inject_mha, "key": inject_mha}
                               }

        injected_clip_model = InjectedClip(clip_model, injected_layers).cuda()

        injected_clip_model.eval()

        if cfg['dataset'] == "imagenet":
            print("Preparing ImageNet dataset.")
            dataset = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

            val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=64, num_workers=4,
                                                     shuffle=False)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=4,
                                                      shuffle=False)

            train_loader = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=4,
                                                       shuffle=True)
        else:
            print("Preparing dataset.")
            dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

            val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False,
                                           tfm=preprocess, shuffle=False)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False,
                                            tfm=preprocess, shuffle=False)

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))
            ])

            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=256,
                                             tfm=train_transform, is_train=True, shuffle=True)

        # we return clip_model here so that it can be used for further experiments
        clip_model = finetune_model(cfg, injected_clip_model, train_loader, val_loader, test_loader, dataset, branch)


if __name__ == '__main__':
    main()
