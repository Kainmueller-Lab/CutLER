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
import torch

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


def get_affinity_matrix(feats, act=set_min, loc=False, sigma=2, sqrt=True):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0, 1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A.astype(np.double)
    A = act(A)
    if loc:
        distance_pen = construct_distant_penalty_matrix(np.sqrt(A.shape[0]), sigma=sigma, sqrt=sqrt, double_sqrt=False)
        A *= distance_pen
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    T = A / np.sum(A, axis=1, keepdims=True)
    return T, A, D

def estimate_left_eig(A, D, subset_by_index=None):
    D_inv = np.diag(1/np.diag(D))
    eigvals, eigvec = eigh(D_inv @ A @ D_inv, D_inv, subset_by_index=subset_by_index)
    return eigvals, eigvec
    
def pcca_torch(T, M, pi):
    # coarse-grained stationary distribution
    pi_coarse = torch.matmul(M.T, pi)
    # HMM output matrix
    B = torch.linalg.multi_dot([torch.diag(1.0 / pi_coarse), M.T, torch.diag(pi)])
    # renormalize B to make it row-stochastic
    B /= B.sum(dim=1, keepdims=True)

    # coarse-grained transition matrix
    # W = torch.linalg.inv(torch.matmul(M.T, M))
    # T_coarse = torch.matmul(W, A)
    A = torch.matmul(torch.matmul(M.T, T), M)
    T_coarse = torch.linalg.solve(torch.matmul(M.T, M), A)
    

    # symmetrize and renormalize to eliminate numerical errors
    X = torch.matmul(torch.diag(pi_coarse), T_coarse)
    # and normalize
    T_coarse = X / X.sum(dim=1, keepdims=True)
    return T_coarse

class Membership(torch.nn.Module):
    def __init__(self, N, M,memberships=None, device='cuda'):
        super().__init__()
        self.membership = torch.nn.Parameter(torch.randn((N,M), device=device))
        if memberships is not None:
            self.set_params(memberships)
        self.acti = torch.nn.Softmax()

    def forward(self, x):
        return x @ self.get_membership()

    def get_membership(self):
        return self.acti(self.membership)
    
    def set_params(self, values):
        with torch.no_grad():
            self.membership=self.membership.copy_(values)

class MembershipEstimator(torch.nn.Module):
    def __init__(
            self, net: Membership, A:np.ndarray, D:np.ndarray, sym=False, lr=0.001, 
            weight_decay=0.1, mode='regularize', epsilon=1e-6, device='cuda'
            ):
        super().__init__()
        self.net = net.to(device)
        self.T = torch.Tensor(A/A.sum(1, keepdims=True)).to(device)
        size = A.shape[0]
        # self.pi = torch.Tensor(estimate_left_eig(A, D, subset_by_index=[size-1,size-1])[1][:,0]).to(device)
        # self.pi = self.pi/self.pi.sum()
        pi = np.diag(D)
        self.pi = torch.Tensor(pi/pi.sum()).to(device)
        self.sym = sym
        self.lr = lr
        self.optimizer = None
        self.mode = mode
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.score = []


    def train(self, epochs, print_every=100):
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.training_step()
            if not epoch%print_every:
                print(loss)
            self.score.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
        M, T_coarse = self.get_results()
        self.M = M.detach().cpu().numpy()
        self.T_coarse = T_coarse.detach().cpu().numpy()
        return self.score
    
    def training_step(self, ) -> torch.Tensor:
        T_coarse = pcca_torch(self.T, self.net.get_membership(), self.pi)
        return -torch.trace(T_coarse)  #+ 5.*(eigenvalue_loss(cxx_b, self.epsilon) + eigenvalue_loss(cyy_b, self.epsilon)) 

    def get_results(self):
        M = self.net.get_membership()
        T_coarse = pcca_torch(self.T, M , self.pi)
        return M, T_coarse
    
    def configure_optimizers(self):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return self.optimizer
    
    def predict(self, loader, noisefree=False):
        
        return loader

    def save(self, path: str):
        r"""Save the current estimator at path.

        Parameters
        ----------
        path: str
            The path where to save the model.

        """
        save_dict = {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
        }
        torch.save(save_dict, path)

    def load(self, path: str):
        r"""Load the estimator from path.
         The architecture needs to fit!

        Parameters
        ----------
        path: str
             The path where the model is saved.
        """

        checkpoint = torch.load(path, map_location=self.device)
        if self.optimizer is None:
            self.configure_optimizers()
        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

PATCHSIZE = 8
VIT_ARCH = 'base'
PRETRAIN_PATH = None
VIT_FEAT = 'k'
CUDA = True
HARD = False

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

    for (names, act) in [['shift', shift]]:  # [['set_min', set_min], ['softmax', softmax], ['softmax_3T', softmax_3T],
                        #  ['softmax_halfT', softmax_halfT], ['shift', shift]]
        n_eigvecs = 8
        n_s = 4*8
        n_figures = n_eigvecs
        figure, axes = plt.subplots(n_s, n_figures, figsize=(n_figures * 4, n_s * 4))
        for i in range(1, 9):
            input_image = f'demo{i}.jpg'
            img_path = 'imgs/' + input_image
            feats = get_feat(img_path=img_path, backbone=backbone, patch_size=PATCHSIZE)
            T, A, D = get_affinity_matrix(feats, act=act, loc=False, sigma=None, sqrt=False)
            size = T.shape[0]
            net=Membership(size,n_eigvecs)
            estimator=MembershipEstimator(net, A, D, weight_decay=0., lr=0.01)
            scores = estimator.train(3000, 1000)
            memberships = estimator.M
            T_coarse = estimator.T_coarse
            eigvals, eigvecs = np.linalg.eig(T_coarse)
            sort_eigs = np.argsort(eigvals)
            eigvals, eigvecs = eigvals[sort_eigs], eigvecs[:,sort_eigs]

            eigvals_origin, eigvecs_right = eigh(A, D, subset_by_index=[size - n_eigvecs, size - 1])
            eigvals_origin_left, eigvecs_left = estimate_left_eig(A, D, subset_by_index=[size - n_eigvecs, size - 1])

            for j in range(n_eigvecs):
                idx_right = (i-1)*4
                idx_left = idx_right+1
                idx_coarse = idx_right + 2
                idx_member = idx_right + 3
                if HARD:
                    sign = np.sign(eigvecs_right[np.argmax(np.abs(eigvecs_right[:,j])),j])
                    e_r = eigvecs_right[:,j] * sign > np.max(np.abs(eigvecs_right[:,j]))*0.2
                    if j==(n_eigvecs-1): # stationary distribution
                        e_l = eigvecs_left[:,j]/eigvecs_left[:,j].sum() # normalize
                        e_l = e_l < 0.9*1/eigvecs_left[:,j].shape[0] # more than random
                    else:
                        sign = np.sign(eigvecs_left[np.argmax(np.abs(eigvecs_left[:,j])),j])
                        e_l = eigvecs_left[:,j] * sign > np.max(np.abs(eigvecs_left[:,j]))*0.2
                else:
                    e_r = eigvecs_right[:,j]
                    e_l = eigvecs_left[:,j]
                    e_coarse = memberships @ eigvecs[:,j]
                axes[idx_right, j].set_title('Eigval: {:.3}'.format(eigvals_origin[j]))
                axes[idx_right, j].imshow(e_r.reshape(60,60))
                axes[idx_left, j].set_title('Eigval: {:.3}'.format(eigvals_origin_left[j]))
                axes[idx_left, j].imshow(e_l.reshape(60,60))
                axes[idx_coarse, j].set_title('Eigval: {:.3}'.format(eigvals[j]))
                axes[idx_coarse, j].imshow(e_coarse.reshape(60,60))
                if j< (n_eigvecs-1):
                    axes[idx_member, j].set_title('Member: {:.3}'.format(eigvals[j]))
                    axes[idx_member, j].imshow(memberships[:,j].reshape(60,60))
                else:
                    axes[idx_member, j].set_title('Member: {:.3}'.format(eigvals[j]))
                    axes[idx_member, j].imshow(np.argmax(memberships, axis=-1).reshape(60,60), cmap='magma')
                
        plt.savefig('./Results/res_eigvecs_NN_{}.jpg'.format(names))


if __name__ == '__main__':
    main()
