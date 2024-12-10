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
from third_party.TokenCut.unsupervised_saliency_detection.object_discovery import detect_box
from third_party.TokenCut.unsupervised_saliency_detection import utils
from utils.matrix import construct_distant_penalty_matrix
# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])



def get_feat(img_path, backbone, patch_size, fixed_size=480, cpu=False):
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    if not cpu: tensor = tensor.cuda()
    feat = backbone(tensor)[0]
    return feat, w, h

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
    return np.exp(0.5*A)

def shift_softmax(A):
    return np.exp((A+1)/2)

def shift(A):
    A = A + 1
    return A/2

def get_affinity_matrix(feats, act=set_min):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A.astype(np.double)
    A=act(A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    T = A/np.sum(A, axis=1, keepdims=True)
    return T, A, D

def get_affinity_matrix_loc(feats, act=set_min, sigma=2, selection=None, sqrt=True, double_sqrt=True):
    n_patches_x = np.sqrt(feats.shape[1])
    distance_pen = construct_distant_penalty_matrix(n_patches_x, sigma=sigma, sqrt=sqrt, double_sqrt=double_sqrt)
    if selection is not None:
        print('using small local pen')
        distance_pen = distance_pen[selection][:,selection]
        feats = feats[:,selection]
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A.astype(np.double)
    A = act(A) * distance_pen
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    T = A / np.sum(A, axis=1, keepdims=True)
    return T, A, D

PATCHSIZE=8
VIT_ARCH='small'
PRETRAIN_PATH=None
VIT_FEAT='k'
CUDA=True
LOCAL=True
SIGMA=8
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
    print (msg)
    backbone.eval()
    if CUDA:
        backbone.cuda()
    N_Iterations=5
    m_list = [2]
    n_s = 2 *  N_Iterations
    n_figures = 8
    figure, axes = plt.subplots(n_s, n_figures, figsize=(n_figures*4,n_s*4))
    for (names, act) in [['shift', shift]]:
        for i in range(1,9):
            input_image = f'demo{i}.jpg'
            img_path = 'imgs/' + input_image
            
            feats, w, h = get_feat(img_path=img_path, backbone=backbone, patch_size=PATCHSIZE)
            selection=np.ones(feats.shape[1], dtype=bool)

            for n_it in range(N_Iterations):
                original_indexes = np.where(selection)
                if LOCAL:
                    T, A, D = get_affinity_matrix_loc(feats, act=act, sigma=SIGMA, selection=selection, sqrt=True)
                else:
                    T, A, D = get_affinity_matrix(feats[:,selection], act=act)
                m=2
                Pcca_m = pcca(T, m)
                T_m = Pcca_m.coarse_grained_transition_matrix
                member = np.zeros((feats.shape[1], 2))
                member[original_indexes] = Pcca_m.memberships
                eigvals, eigvec = np.linalg.eig(T_m)
                sort_id = np.argsort(eigvals)
                second_larg_id = sort_id[-2]
                id_max = np.argmax(np.abs(eigvec[:, second_larg_id]))
                highest_prob = np.argmax(member[:,id_max])
                binary = member[:,id_max]>0.5
                y_h = highest_prob//60
                x_h = highest_prob%60
                x_im = n_it*2
                axes[x_im,i-1].set_title('Eigval: {:.3}'.format(eigvals[second_larg_id]))
                axes[x_im,i-1].plot(x_h, y_h, '.', color='r')
                dims = (60,60)
                bipartition = binary.reshape(dims).astype(float)
                axes[x_im,i-1].imshow(bipartition)
                _, _, _, cc = detect_box(bipartition, highest_prob, dims, scales=[PATCHSIZE, PATCHSIZE], initial_im_size=[h,w])
                pseudo_mask = np.zeros(dims)
                pseudo_mask[cc[0],cc[1]] = 1
                axes[x_im+1,i-1].imshow(pseudo_mask)
                axes[x_im+1,i-1].plot(x_h, y_h, '.', color='r')
                # remove the binary one
                selection = np.logical_and(np.logical_not(pseudo_mask.flatten()), selection)
                

            # T, A, D = get_affinity_matrix(feats, act=act)
            # size = T.shape[0]
            # eigvals_origin, _ = eigh(A, D, subset_by_index=[size-max(m_list)-4,size-2])
            
            # # axes[0,-1].set_title('Eivals')
            # # axes[0,-1].plot(eigvals_origin, '.')
            # # axes[0,-1].set_yscale('log')
            # m=2
            # Pcca_m = pcca(T, m)
            # T_m = Pcca_m.coarse_grained_transition_matrix
            # member = Pcca_m.memberships
            # eigvals, eigvec = np.linalg.eig(T_m)
            # sort_id = np.argsort(eigvals)
            # second_larg_id = sort_id[-2]
            # id_max = np.argmax(np.abs(eigvec[:, second_larg_id]))
            # highest_prob = np.argmax(member[:,id_max])
            # binary = member[:,id_max]>0.5
            # y_h = highest_prob//60
            # x_h = highest_prob%60
            # axes[0,i-1].set_title('Eigval: {:.3}'.format(eigvals[second_larg_id]))
            # axes[0,i-1].plot(x_h, y_h, '.', color='r')
            # dims = (60,60)
            # bipartition = binary.reshape(dims).astype(float)
            # axes[0,i-1].imshow(bipartition)
            # _, _, _, cc = detect_box(bipartition, highest_prob, dims, scales=[PATCHSIZE, PATCHSIZE], initial_im_size=[h,w])
            # pseudo_mask = np.zeros(dims)
            # pseudo_mask[cc[0],cc[1]] = 1
            # axes[1,i-1].imshow(pseudo_mask)
            # axes[1,i-1].plot(x_h, y_h, '.', color='r')
            # # remove the binary one
            # left = np.logical_not(pseudo_mask.flatten())
            # original_indexes = np.where(left)
            # T_new, A_new, D_new = get_affinity_matrix(feats[:,left], act=act)
            # Pcca_m = pcca(T_new, m)
            # T_m = Pcca_m.coarse_grained_transition_matrix
            # member_new = np.zeros_like(member)
            # member_new[original_indexes] = Pcca_m.memberships
            # eigvals, eigvec = np.linalg.eig(T_m)
            # sort_id = np.argsort(eigvals)
            # second_larg_id = sort_id[-2]
            # id_max = np.argmax(np.abs(eigvec[:, second_larg_id]))
            # highest_prob = np.argmax(member_new[:,id_max])
            # binary = member_new[:,id_max]>0.5

            # bipartition = binary.reshape(dims).astype(float)
            # axes[2,i-1].imshow(bipartition.reshape(60,60))
            # _, _, _, cc = detect_box(bipartition, highest_prob, dims, scales=[PATCHSIZE, PATCHSIZE], initial_im_size=[h,w])
            # pseudo_mask = np.zeros(dims)
            # pseudo_mask[cc[0],cc[1]] = 1
            # axes[2,i-1].set_title('Eigval: {:.3}'.format(eigvals[second_larg_id]))
            # axes[3,i-1].imshow(pseudo_mask.reshape(60,60))
            # # axes[1,i-1].imshow(binary.reshape(60,60))
            # y_h = highest_prob//60
            # x_h = highest_prob%60
            # axes[2,i-1].plot(x_h, y_h, '.', color='r')
            # axes[3,i-1].plot(x_h, y_h, '.', color='r')


    plt.savefig('./Results/res_iterative_{}_{}.jpg'.format(N_Iterations, names))
            
    

if __name__=='__main__':
    main()