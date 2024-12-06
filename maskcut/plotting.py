import dino
from torchvision import transforms
import PIL
import PIL.Image as Image
import numpy as np
from deeptime.markov import pcca
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.linalg import eigh


sys.path.append('../')
from third_party.TokenCut.unsupervised_saliency_detection import utils
from utils.matrix import construct_distant_penalty_matrix


# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225)), ])


def get_feat(img_path, backbone, patch_size, fixed_size=480, cpu=False):
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]
    return feat


def set_min(A, tau=0, eps=1e-5):
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    return A


def softmax(A):
    A = np.exp(A)
    return A


def softmax_3T(A):
    return np.exp(A * 3)


def softmax_halfT(A):
    return np.exp(0.5 * A)


def shift(A):
    A = A + 1
    return A / 2


def get_affinity_matrix(feats, act=set_min, sigma=2, sqrt=True, double_sqrt=True):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A.astype(np.double)
    distance_pen = construct_distant_penalty_matrix(np.sqrt(A.shape[0]), sigma=sigma, sqrt=sqrt, double_sqrt=double_sqrt)
    A = act(A) * distance_pen
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    T = A / np.sum(A, axis=1, keepdims=True)
    return T, A, D


PATCHSIZE = 8
VIT_ARCH = 'small'
PRETRAIN_PATH = None
VIT_FEAT = 'k'
CUDA = True


def main():
    if VIT_ARCH == 'base' and PATCHSIZE == 8:
        if PRETRAIN_PATH is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif VIT_ARCH == 'small' and PATCHSIZE == 8:
        if PRETRAIN_PATH is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384
    backbone = dino.ViTFeat(url, feat_dim, VIT_ARCH, VIT_FEAT, PATCHSIZE)
    msg = 'Load {} pre-trained feature...'.format(VIT_ARCH)
    print(msg)
    backbone.eval()
    if CUDA:
        backbone.cuda()

    for (names, act) in [['softmax', softmax]]:  # [['set_min', set_min], ['softmax', softmax], ['softmax_3T', softmax_3T],
                        #  ['softmax_halfT', softmax_halfT], ['shift', shift]]
        for i in range(1, 9):
            input_image = f'demo{i}.jpg'
            img_path = 'imgs/' + input_image
            m_list = [2, 3, 5, 7, 10]
            n_s = len(m_list)
            n_figures = max(m_list) + 1
            feats = get_feat(img_path=img_path, backbone=backbone, patch_size=PATCHSIZE)
            for sigma in [45,50,55,60,65,70,75,80,85,90,95,100]: # [4,8,10,12]
                T, A, D = get_affinity_matrix(feats, act=act, sigma=sigma, sqrt=False, double_sqrt=False)
                size = T.shape[0]
                eigvals_origin, _ = eigh(A, D, subset_by_index=[size - max(m_list) - 4, size - 2])
                figure, axes = plt.subplots(n_s, n_figures, figsize=(n_figures * 4, n_s * 4))
                axes[0, 2].set_title('Eivals')
                axes[0, 2].plot(eigvals_origin, '.')
                axes[0, 2].set_yscale('log')
                for m_i, m in enumerate(m_list):
                    Pcca_m = pcca(T, m)
                    T_m = Pcca_m.coarse_grained_transition_matrix
                    member = Pcca_m.memberships
                    eigvals, eigvec = np.linalg.eig(T_m)
                    sort_id = np.argsort(eigvals)
                    second_larg_id = sort_id[-2]
                    sort_second = np.argsort(eigvec[:, second_larg_id])

                    for ind_i, ind in enumerate(sort_second):
                        axes[m_i, ind_i].set_title('Eigvec: {:.3}'.format(eigvec[ind, second_larg_id]))
                        axes[m_i, ind_i].imshow(member[:, ind].reshape(60, 60))
                    hard_assignment = np.argmax(member, axis=-1)
                    axes[m_i, -1].imshow(hard_assignment.reshape(60,60))
                plt.savefig('./Results/res_{}_{}_{}'.format(names, sigma, input_image))


if __name__ == '__main__':
    main()
