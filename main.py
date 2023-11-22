import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F


def apply_gaussian_blur(image, kernel, kernel_size):
    # Expand dimensions for convolution
    image = image.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Apply convolution with Gaussian kernel
    blurred_image = F.conv2d(image, kernel, padding=kernel_size//2)

    # Remove the extra batch dimension and return the blurred image
    return blurred_image.squeeze(0)


def create_gaussian_kernel(kernel_size, sigma):
    # Create a 1D Gaussian kernel
    kernel_1d = torch.exp(-(torch.arange(kernel_size) - kernel_size // 2) ** 2 / (2 * sigma ** 2))

    # Normalize the kernel
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Expand dimensions to create a 2D kernel
    kernel_2d = torch.outer(kernel_1d, kernel_1d)

    return kernel_2d


def H(phi):
    return 0.5 * (1 + 2 * torch.arctan(phi / 0.1) / torch.pi)


def d_dx(f):
    return f[:, 1:] - f[:, :-1]


def d_dy(f):
    return f[1:, :] - f[:-1, :]


def get_pixel_values(image, coordinates=None):
    # Expand dimensions for grid_sample
    image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    if coordinates is None:
        # Create coordinate grid
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, image.shape[2]), torch.linspace(-1, 1, image.shape[3]))
        grid = torch.stack((grid_y, grid_x), dim=2).unsqueeze(0)

    else:
        # Normalize coordinates to [-1, 1]
        normalized_coords = coordinates * 2 - 1
        # normalized_coords = coordinates

        # Reshape the normalized coordinates to match the image size
        grid = normalized_coords.view(1, image.shape[2], image.shape[3], 2)

    # Perform grid_sample with bilinear interpolation
    interpolated_values = F.grid_sample(image, grid.to(image.device), mode='bilinear', padding_mode='border')
    return interpolated_values


def grad(image, coords=None):
    # # Expand dimensions for grid_sample
    # image = image.unsqueeze(0)
    # if len(image.shape) == 3:
    #     image = image.unsqueeze(0)
    #
    # if coords is None:
    #     # Create coordinate grid
    #     grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, image.shape[2]), torch.linspace(-1, 1, image.shape[3]))
    #     grid = torch.stack((grid_y, grid_x), dim=2).unsqueeze(0)
    # else:
    #     # Normalize coordinates to [-1, 1]
    #     normalized_coords = coords * 2 - 1
    #
    #     # Reshape the normalized coordinates to match the image size
    #     grid = normalized_coords.view(1, image.shape[2], image.shape[3], 2)
    #
    # # Perform grid_sample with bilinear interpolation
    # interpolated_values = F.grid_sample(image, grid.to(image.device), mode='bilinear', padding_mode='border')
    interpolated_values = get_pixel_values(image, coords)

    # Compute gradients using central differences
    gradients_x = (interpolated_values[:, :, 2:, :] - interpolated_values[:, :, :-2, :]) / 2
    gradients_y = (interpolated_values[:, :, :, 2:] - interpolated_values[:, :, :, :-2]) / 2

    # pad with zeros to stay in the same shape:
    gradients_x = F.pad(gradients_x, (0, 0, 1, 1), mode='constant', value=0)
    gradients_y = F.pad(gradients_y, (1, 1, 0, 0), mode='constant', value=0)

    return torch.cat([gradients_x.squeeze(0), gradients_y.squeeze(0)])


class EoccLoss(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        weights = torch.linspace(0, 1, epochs)
        self.t = 100 * torch.exp(-weights * torch.log(torch.tensor([100 / 1])))  # 5 * torch.exp(-weights * torch.log(torch.tensor([5 / 1])))
        self.ni = 0.01
        # gaussian_kernel = transforms.GaussianBlur(kernel_size=5, sigma=0.1)  # size is not mentioned in the paper.
        # self.G = gaussian_kernel.weight.unsqueeze(0).unsqueeze(0)
        self.kernel_size = 21  # size is not mentioned in the paper.
        sigma = 2  # in the paper, sigma=0.1. but this has no effect.
        self.G = create_gaussian_kernel(kernel_size=self.kernel_size, sigma=sigma)
        self.G = self.G.to(device)

    def forward(self, e_occ, phi, epoch):
        # Eq. 19
        e_occ_map = e_occ.view(im_h, im_w)  # TODO: check correctness of stacking
        phi = phi.view(im_h, im_w)  # TODO: check correctness of stacking
        e_occ_blurred = apply_gaussian_blur(e_occ_map, self.G, kernel_size=self.kernel_size)
        H_phi = H(phi)
        out_map = e_occ_blurred * H_phi + self.t[epoch] * (1 - H_phi) + self.ni * grad(H_phi).norm(p=1, dim=0)
        return out_map.mean()  # TODO: consider .sum(),


class ELLoss(nn.Module):
    def __init__(self, im_l, im_r):
        super().__init__()
        self.eta = 0.001
        self.im_l = im_l
        self.im_r = im_r
        self.lambda_ = 2
        self.beta = 1  # TODO: I dont know what should be the value here
        self.alpha_hat = 100.9375 * self.beta  # paper: 0.9375 * self.beta
        self.eps_hat = 0.25 * self.beta

    # def forward(self, g_l, xy, d_l, phi_l, v_l):
    #     # Eq. 23
    #
    #     # compute e_d
    #
    #     # plt.imshow(g_l[:, 0].view(im_h, im_w).detach()), plt.colorbar(), plt.show()
    #
    #     Ir_gl = get_pixel_values(self.im_r, g_l)
    #     # plt.imshow(Ir_gl[0].detach().permute(1, 2, 0)), plt.show()
    #     Il_xy_l = get_pixel_values(self.im_l, xy)
    #     # plt.imshow(Il_xy_l[0].detach().permute(1, 2, 0)), plt.show()
    #     grad_Ir_gl = grad(self.im_r, g_l)  # TODO: if takes runtime, consider calculating the grads images in advance
    #     # plt.imshow(grad_Ir_gl[0].detach()), plt.show()
    #     grad_Il_xy_l = grad(self.im_l, xy)
    #     sd_2 = (Ir_gl - Il_xy_l).norm(p=1) ** 2 + self.lambda_ * (grad_Ir_gl - grad_Il_xy_l).norm(p=1) ** 2
    #     e_d = (sd_2 + self.eta ** 2).sqrt()
    #
    #     # compute e_s
    #     grad_d_l = grad(d_l.view(im_h, im_w))
    #     e_s = self.beta * (grad_d_l.norm(p=1) ** 2 + self.eta ** 2).sqrt()
    #
    #     v_l = v_l.view(im_h, im_w)
    #     grad_v_l = grad(v_l)
    #     phi_l = phi_l.view(im_h, im_w)
    #
    #     out_map = (e_d + e_s * v_l ** 2 + self.alpha_hat * (v_l - 1) ** 2) * H(phi_l) + self.eps_hat * grad_v_l.norm(p=1) ** 2
    #     return out_map.mean(), e_s, e_d

    def forward(self, g, xy, d, phi, v, im_side):
        if im_side == 'left':
            im_src = self.im_l
            im_ref = self.im_r
        else:
            im_src = self.im_r
            im_ref = self.im_l
        I1_xy = get_pixel_values(im_src, xy)
        # plt.imshow(I1_xy[0].detach().permute(1, 2, 0)), plt.show()
        I2_g = get_pixel_values(im_ref, g)
        # plt.imshow(I2_g[0].detach().permute(1, 2, 0)), plt.show()
        grad_I1_xy = grad(im_src, xy)
        grad_I2_g = grad(im_ref, g_l)  # TODO: if takes runtime, consider calculating the grads images in advance
        # plt.imshow(grad_I2_g[0].detach()), plt.show()

        sd_2 = (I2_g - I1_xy).norm(p=1) ** 2 + self.lambda_ * (grad_I2_g - grad_I1_xy).norm(p=1) ** 2
        e_d = (sd_2 + self.eta ** 2).sqrt()

        # compute e_s
        grad_d = grad(d.view(im_h, im_w))
        e_s = self.beta * (grad_d.norm(p=1) ** 2 + self.eta ** 2).sqrt()

        v = v.view(im_h, im_w)
        grad_v = grad(v)
        phi = phi.view(im_h, im_w)

        out_map = (e_d + e_s * v ** 2 + self.alpha_hat * (v - 1) ** 2) * H(phi) + self.eps_hat * grad_v.norm(
            p=1) ** 2
        return out_map.mean(), e_s, e_d




class MyData(Dataset):
    def __init__(self, image_left, image_right, disparity_left, disparity_right):
        super().__init__()
        self.image_left = np.array(Image.open(image_left), dtype='float32')
        self.image_right = np.array(Image.open(image_right), dtype='float32')
        self.disparity_left = np.array(Image.open(disparity_left), dtype='float32').flatten()/4
        self.disparity_left = self.disparity_left / self.image_left.shape[1]  # convert from pixel coords to [0, 1]
        self.disparity_right = np.array(Image.open(disparity_right), dtype='float32').flatten()/4
        self.disparity_right = self.disparity_right / self.image_right.shape[1]
        x, y = np.meshgrid(range(self.image_left.shape[1]), range(self.image_left.shape[0]))
        x, y = x.astype('float32'), y.astype('float32')
        self.xy = np.stack([x.flatten() / self.image_left.shape[1], y.flatten() / self.image_left.shape[0]], axis=1)

        # self.disparity_right = self.disparity_right * 2 - 1
        # self.disparity_left = self.disparity_left * 2 - 1
        # self.xy = self.xy * 2 - 1
        self.to_tensor = transforms.ToTensor()

        # start from right and get the left
        # d = torch.stack([batch['xy'][:, 0] - batch['d_l'], batch['xy'][:, 1]], dim=1)
        # a = get_pixel_values(loss2.im_r, d)
        # plt.imshow(a[0].detach().permute(1, 2, 0)), plt.show()

        # start from left and get the right
        # d = torch.stack([batch['xy'][:, 0] + batch['d_r'], batch['xy'][:, 1]], dim=1)
        # a = get_pixel_values(loss2.im_l, d)
        # plt.imshow(a[0].detach().permute(1, 2, 0)), plt.show()

    def __len__(self):
        return len(self.xy)

    def __getitem__(self, idx):
        # TODO d = disparity_right?
        # TODO x, y in [0,1]?
        xy = torch.from_numpy(self.xy[idx])
        return {'xy': xy, 'd_l': self.disparity_left[idx], 'd_r': self.disparity_right[idx]}


class Phi(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1))

        # self.model = nn.Linear(2, 1)

    def forward(self, xy):
        return self.model(xy)


class Disparity(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 1), nn.Sigmoid())

        # self.model = nn.Linear(2, 1)

    def forward(self, xy):
        return self.model(xy)


class Pipeline(nn.Module):
    def __init__(self, eps_occl, hidden_dim=10):
        super().__init__()
        self.eps_occl = eps_occl
        self.hidden_dim = hidden_dim
        self.phi_l = Phi(hidden_dim)
        self.phi_r = Phi(hidden_dim)
        self.d_l_v = Disparity(hidden_dim)
        self.d_r_v = Disparity(hidden_dim)
        self.d_l_o = Disparity(hidden_dim)
        self.d_r_o = Disparity(hidden_dim)
        self.v_l = nn.Sequential(nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.v_r = nn.Sequential(nn.Linear(2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), nn.Sigmoid())

    # def forward(self, batch):
    #     phi_l_xy_l = self.phi_l(batch['xy'])
    #     H_l = H(phi_l_xy_l)
    #     dl = self.d_l_v(batch['xy']) * H_l + \
    #           self.d_l_o(batch['xy']) * (1 - H_l)
    #
    #     gl = torch.cat([batch['xy'][:, [0]] - dl, batch['xy'][:, [1]]], dim=1)
    #     # g_l = torch.cat([dl, batch['xy'][:, [1]]], dim=1)
    #     H_r = H(self.phi_r(gl))
    #     dr_gl = self.d_r_v(gl) * H_r + \
    #           self.d_r_o(gl) * (1 - H_r)
    #
    #     # plt.imshow(dl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
    #     # plt.imshow(dr_gl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
    #     ul = dl + (-dr_gl)  # different from the paper since here the base corrds are the same (xy_l == xy_r)
    #     e_occ_l = -torch.log(self.eps_occl + (1 - self.eps_occl) * torch.exp(-torch.abs(ul)))
    #
    #     v_l = self.v_l(batch['xy'])
    #     return e_occ_l, g_l, phi_l_xy_l, d_l, v_l

    def forward(self, batch):
        return self.calculate_aux(batch)

    def calculate(self, xy, phi1, d_v1, d_o1, phi2, d_v2, d_o2, im_side):
        phi_xy = phi1(xy)
        H_phi = H(phi_xy)
        d = d_v1(xy) * H_phi + \
             d_o1(xy) * (1 - H_phi)

        if im_side == 'left':
            g = torch.cat([xy[:, [0]] - d, xy[:, [1]]], dim=1)
        else:
            g = torch.cat([xy[:, [0]] + d, xy[:, [1]]], dim=1)

        # g_l = torch.cat([dl, batch['xy'][:, [1]]], dim=1)
        H_g = H(phi2(g))
        dg = d_v2(g) * H_g + \
                d_o2(g) * (1 - H_g)

        # plt.imshow(d.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        # plt.imshow(dg.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        u = d + (-dg)  # different from the paper since here the base corrds are the same (xy_l == xy_r)
        e_occ = -torch.log(self.eps_occl + (1 - self.eps_occl) * torch.exp(-torch.abs(u)))

        return e_occ, g, phi_xy, d

    def calculate_aux(self, batch):
        e_occ_l, gl, phi_l_xy, dl = self.calculate(batch['xy'], self.phi_l, self.d_l_v, self.d_l_o,
                                                   self.phi_r, self.d_r_v, self.d_r_o, 'left')
        e_occ_r, gr, phi_r_xy, dr = self.calculate(batch['xy'], self.phi_r, self.d_r_v, self.d_r_o,
                                                   self.phi_l, self.d_l_v, self.d_l_o, 'right')
        vl = self.v_l(batch['xy'])
        vr = self.v_r(batch['xy'])

        return e_occ_l, gl, phi_l_xy, dl, vl, e_occ_r, gr, phi_r_xy, dr, vr

    def forward2(self, batch):
        phi_l_xy = self.phi_l(batch['xy'])
        Hl = H(phi_l_xy)
        dl = self.d_l_v(batch['xy']) * Hl + \
              self.d_l_o(batch['xy']) * (1 - Hl)

        gl = torch.cat([batch['xy'][:, [0]] - dl, batch['xy'][:, [1]]], dim=1)
        # g_l = torch.cat([dl, batch['xy'][:, [1]]], dim=1)
        Hr_gl = H(self.phi_r(gl))
        dr_gl = self.d_r_v(gl) * Hr_gl + \
              self.d_r_o(gl) * (1 - Hr_gl)

        # plt.imshow(dl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        # plt.imshow(dr_gl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        ul = dl + (-dr_gl)  # different from the paper since here the base corrds are the same (xy_l == xy_r)
        e_occ_l = -torch.log(self.eps_occl + (1 - self.eps_occl) * torch.exp(-torch.abs(ul)))

        vl = self.v_l(batch['xy'])

        ##############################################################
        # Symmetric direction:
        ##############################################################
        phi_r_xy = self.phi_r(batch['xy'])
        Hr = H(phi_r_xy)
        dr = self.d_r_v(batch['xy']) * Hr + \
             self.d_r_o(batch['xy']) * (1 - Hr)

        gr = torch.cat([batch['xy'][:, [0]] + dr, batch['xy'][:, [1]]], dim=1)
        # g_l = torch.cat([dl, batch['xy'][:, [1]]], dim=1)
        Hl_gr = H(self.phi_l(gr))
        dl_gr = self.d_l_v(gr) * Hl_gr + \
                self.d_l_o(gr) * (1 - Hl_gr)

        # plt.imshow(dl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        # plt.imshow(dr_gl.view(im_h, im_w).detach()), plt.colorbar(), plt.show()
        ur = dr + (-dl_gr)  # different from the paper since here the base corrds are the same (xy_l == xy_r)
        e_occ_r = -torch.log(self.eps_occl + (1 - self.eps_occl) * torch.exp(-torch.abs(ur)))

        vr = self.v_r(batch['xy'])

        return e_occ_l, e_occ_r, gl, gr, phi_l_xy, phi_r_xy, dl, dr, vl, vr


######
# im_l(xy + d_r) == im_r
# im_r(xy_r - d_l) == im_l
######


seed = 7
torch.manual_seed(seed)
np.random.seed(seed)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
data_name = 'teddy'  # 'cones'  # 'teddy'


dataset = MyData(f'dataset/{data_name}/im_l.png', f'dataset/{data_name}/im_r.png',
                 f'dataset/{data_name}/disp_l.png', f'dataset/{data_name}/disp_r.png')

im_w, im_h = 450, 375
batch_size = im_w * im_h
epochs = 10000
lr = 1
hidden_dim = 10

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
im_l = Image.open(f'dataset/{data_name}/im_l.png')
im_r = Image.open(f'dataset/{data_name}/im_r.png')
to_tensor = transforms.ToTensor()

eps_occl = 0.005  # 0.005 # 0.01 # 0.02
pipe = Pipeline(eps_occl=eps_occl, hidden_dim=hidden_dim).to(device)

optim = torch.optim.SGD(pipe.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [100, 500, 1000], 0.5)
loss1 = EoccLoss(epochs)
loss2 = ELLoss(to_tensor(im_l), to_tensor(im_r))

best_loss = torch.inf
for epoch in range(epochs):
    for batch in data_loader:
        xy = batch['xy'].to(device)

        # update phi:
        e_occ_l, g_l, phi_l, d_l, v_l, e_occ_r, g_r, phi_r, d_r, v_r = pipe(batch)

        # using GT:
        # d_l = batch['d_l'].unsqueeze(1)
        # g_l = torch.cat([xy[:, [0]] - d_l, xy[:, [1]]], dim=1)
        # d_r = batch['d_r'].unsqueeze(1)
        # g_r = torch.cat([xy[:, [0]] + d_r, xy[:, [1]]], dim=1)

        plt.imshow(d_l.detach().view(im_h, im_w), cmap='gray'), plt.title('d_l'), plt.show()
        plt.imshow(d_r.detach().view(im_h, im_w), cmap='gray'), plt.title('d_r'), plt.show()

        l1_l = loss1(e_occ_l, phi_l, epoch)
        l1_r = loss1(e_occ_r, phi_r, epoch)
        l2_l, es_l, ed_l = loss2(g_l, xy, d_l, phi_l, v_l, 'left')
        l2_r, es_r, ed_r = loss2(g_r, xy, d_r, phi_r, v_r, 'right')
        print(f'Ep: {epoch}\tL1-L: {l1_l:.2f}\tL1-R: {l1_r:.2f}\t'
              f'L2-L: {l2_l:.2f}\tL2-R: {l2_r:.2f}\t'
              f'es_l: {es_l:.2f}\tes_r: {es_r:.2f}\t'
              f'ed_l: {ed_l:.0f}\ted_r: {ed_r:.0f}\t'
              f'phi_l: {phi_l.mean():.2f}\tphi_r: {phi_r.mean():.2f}\t'
              f'd_l: {d_l.mean():.2f}\td_r: {d_r.mean():.2f}'
              )

        # if epoch % 10 == 0:
        #     loss = l2 / 50
        # else:
        #     loss = l1

        # loss = l2
        # loss = l1 + 0.0000385 * l2
        # loss = l1 + l2
        loss = l1_l + l1_r + (l2_l + l2_r) / 1000

        # if epoch < 100:
        #     loss = l1
        # else:
        #     loss = l1 + l2


        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

    if epoch > 0 and epoch % 1000 == 0 and loss < best_loss:
        best_loss = loss
        torch.save(pipe.state_dict(), f'checkpoints/model/best.pt')
        torch.save(optim.state_dict(), f'checkpoints/optim/best.pt')
        torch.save(scheduler.state_dict(), f'checkpoints/scheduler/best.pt')

    # if epoch % 200 == 0:
    #     plt.imshow(d_l.detach().cpu().view(im_h, im_w)), plt.show()



