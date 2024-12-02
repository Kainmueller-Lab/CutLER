#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import torch.nn.functional as F
import PIL
import PIL.Image as Image
from deeptime.markov import pcca

sys.path.append('../')
sys.path.append('../third_party')
from maskcut import get_masked_affinity_matrix, get_affinity_matrix, second_smallest_eigenvector, get_salient_areas, \
    check_num_fg_corners
from TokenCut.unsupervised_saliency_detection import utils, metric
from TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
import argparse
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms
from scipy import ndimage

try:
    detectron2_path = os.path.expanduser('~/Documents/detectron2')  # Expands the `~` to the full home directory path
    if detectron2_path not in sys.path:
        sys.path.append(detectron2_path)
except (ImportError, FileNotFoundError) as e:
    pass

from detectron2.utils.colormap import random_color

import dino  # model
from third_party.TokenCut.unsupervised_saliency_detection import metric
from crf import densecrf
from maskcut import maskcut

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225)), ])


def vis_mask(input, mask, mask_color):
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)


def custom_get_affinity_matrix(feats, tau):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()

    # A = A > tau
    A = np.exp(A*3)

    A = A.astype(np.double)

    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    T = A/np.sum(A, axis=1, keepdims=True)

    return A, D, T


def custom_maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    """
        Implementation of MaskCut.
        Inputs
          feats: the pixel/patche features of an image
          dims: dimension of the map from which the features are used
          scales: from image to map scale
          init_image_size: size of the image
          tau: thresold for graph construction
          N: number of pseudo-masks per image.
        """
    bipartitions = []
    eigvecs = []

    # construct the affinity matrix
    A, D, T = custom_get_affinity_matrix(feats, tau)
    N=2
    # get the PCCA membership
    pcca_m = pcca(T, N)
    bipartitions = pcca_m.memberships.T

    # iterate over the bipartitions
    for i in range(N):
        # get the second smallest eigenvector
        #eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)

        bipartition = bipartitions[i] # get_salient_areas(second_smallest_vec)

        # get the index of the second smallest eigenvector
        index_sec_smallest = np.argmax(np.abs(bipartition))

        # make it binary
        bipartition_binary = bipartition.copy() > 1/N

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        nc = check_num_fg_corners(bipartition_binary, dims)
        reverse = False
        if nc >= 3:
            reverse = True

        if reverse:
            # reverse bipartition
            bipartition = 1 - bipartition #
            bipartition_binary = bipartition.copy() > 1/N
            index_sec_smallest = np.argmax(bipartition)

        # get pxiels corresponding to the seed
        bipartition_binary = bipartition_binary.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition_binary, index_sec_smallest, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0], cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size,
                                    mode='nearest').squeeze()
        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = bipartition.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return index_sec_smallest, bipartitions, eigvecs


def custom_maskcut(img_path, backbone, patch_size, tau, N=1, fixed_size=480, cpu=False):
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]

    _, bipartition, eigvec = custom_maskcut_forward(
        feat, [feat_h, feat_w], [patch_size, patch_size], [h, w], tau, N=N,
        cpu=cpu
    )

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, I_new



if __name__ == "__main__":
    parser = argparse.ArgumentParser('MaskCut Demo')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')
    parser.add_argument('--img-path', type=str, default='imgs/demo1.jpg', help='single image visualization')
    parser.add_argument('--tau', type=float, default=0.15, help='threshold used for producing binary graph')

    # additional arguments
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--output_path', type=str, default='Results/', help='path to save outputs')

    args = parser.parse_args()
    print(args)

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)

    msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
    print(msg)
    backbone.eval()
    if not args.cpu:
        backbone.cuda()

    bipartitions, _, I_new = custom_maskcut(args.img_path, backbone, args.patch_size, args.tau, \
                                     N=args.N, fixed_size=args.fixed_size, cpu=args.cpu)

    I = Image.open(args.img_path).convert('RGB')
    width, height = I.size
    pseudo_mask_list = []
    for idx, bipartition in enumerate(bipartitions):
        # post-process pesudo-masks with CRF
        pseudo_mask = densecrf(np.array(I_new), bipartition)
        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

        # filter out the mask that have a very different pseudo-mask after the CRF
        if not args.cpu:
            mask1 = torch.from_numpy(bipartition).cuda()
            mask2 = torch.from_numpy(pseudo_mask).cuda()
        else:
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo_mask)
        if metric.IoU(mask1, mask2) < 0.5:
            pseudo_mask = pseudo_mask * -1

        # construct binary pseudo-masks
        pseudo_mask[pseudo_mask < 0] = 0
        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
        pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower
        pseudo_mask_list.append(pseudo_mask)

    input = np.array(I)
    for pseudo_mask in pseudo_mask_list:
        input = vis_mask(input, pseudo_mask, random_color(rgb=True))
    os.makedirs(args.output_path, exist_ok=True)
    input.save(os.path.join(args.output_path, "demo.jpg"))

