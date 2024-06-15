from tqdm import tqdm

import torch
from torch.utils import checkpoint
import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def clip_classifier_with_grad(classnames, template, clip_model):
    clip_weights = []

    for classname in classnames:
        # Tokenize the prompts
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        texts = clip.tokenize(texts).cuda()
        # prompt ensemble for ImageNet
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings_normed = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding_mean = class_embeddings_normed.mean(dim=0)
        class_embedding_mean_normed = class_embedding_mean / class_embedding_mean.norm()

        # Checkpoint can be used if there is not enough space on the GPU to store the entire back prop graph
        #class_embedding_mean_normed = checkpoint.checkpoint(text_embeddings_forward, *(clip_model, texts), use_reentrant=False)
        clip_weights.append(class_embedding_mean_normed)

    clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def text_embeddings_forward(clip_model, texts):
    # prompt ensemble for ImageNet
    class_embeddings = clip_model.encode_text(texts)
    class_embeddings_normed = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    class_embedding_mean = class_embeddings_normed.mean(dim=0)
    class_embedding_mean_normed = class_embedding_mean / class_embedding_mean.norm()
    return class_embedding_mean_normed

def clip_classifier_per_template(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for t in template:
            clip_weights_by_prompt = []
            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                text = [t.format(classname)]
                text = clip.tokenize(text).cuda()
                class_embedding = clip_model.encode_text(text)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                class_embedding = class_embedding.squeeze()
                clip_weights_by_prompt.append(class_embedding)
            clip_weights_by_prompt = torch.stack(clip_weights_by_prompt, dim=1).cuda()
            clip_weights.append(clip_weights_by_prompt)

        clip_weights = torch.stack(clip_weights, dim=2).cuda()
    return clip_weights


def load_features(clip_model, loader):
    features, labels = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)

    return features, labels


def scale_(x, target):
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()
    
    return y
