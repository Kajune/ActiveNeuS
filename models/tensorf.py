import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel_1d(kernel_size, sigma):
    k_size = kernel_size // 2
    x = torch.arange(-k_size, k_size + 1, dtype=torch.float32)
    distance_squared = x**2
    kernel = torch.exp(-distance_squared / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def gaussian_kernel_2d(kernel_size, sigma):
    k_size = kernel_size // 2
    x = torch.arange(-k_size, k_size + 1, dtype=torch.float32)
    y = torch.arange(-k_size, k_size + 1, dtype=torch.float32)
    grid = torch.meshgrid(x, y)
    distance_squared = grid[0]**2 + grid[1]**2
    kernel = torch.exp(-distance_squared / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def apply_gaussian_filter_1d(input_tensor, kernel_size, sigma):
    kernel = gaussian_kernel_1d(kernel_size, sigma)
    channels = input_tensor.shape[1]
    kernel = kernel.view(1, 1, kernel_size).expand(channels, 1, -1).to(input_tensor)
    filtered_tensor = F.conv1d(input_tensor, kernel, padding=kernel_size//2, groups=channels)
    return filtered_tensor


def apply_gaussian_filter_2d(input_tensor, kernel_size, sigma):
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    channels = input_tensor.shape[1]
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(channels, 1, -1, -1).to(input_tensor)
    filtered_tensor = F.conv2d(input_tensor, kernel, padding="same", groups=channels)
    return filtered_tensor
    


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class TensorBase(nn.Module):
    def __init__(self, aabb, gridSize, density_n_comp=[16,16,16], step_ratio=2.0):
        super().__init__()

        self.density_n_comp = density_n_comp
        self.register_buffer('aabb', torch.from_numpy(aabb))
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.init_svd_volume(gridSize[0])

        self.encoding_dim = sum(density_n_comp)


    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        aabbSize = self.aabb[1] - self.aabb[0]
        self.gridSize = torch.LongTensor(gridSize)
        self.units =aabbSize / (self.gridSize-1)
        self.stepSize =torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(aabbSize)))
        self.nSamples =int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)


    def init_svd_volume(self, res):
        pass


    def forward(self, xyz_sampled):
        pass

    
    def normalize_coord(self, xyz_sampled):
        aabbSize = self.aabb[1] - self.aabb[0]
        invaabbSize = 2.0 / aabbSize
        return (xyz_sampled - self.aabb[0]) * invaabbSize - 1


    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'step_ratio': self.step_ratio,
        }


    def shrink(self, new_aabb, voxel_size):
        pass


    def update(self):
        pass



class TensorVMSplit(TensorBase):
    def init_svd_volume(self, res):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1)


    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total


    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line)


    def smoothness(self, kernel_size=3, sigma=1.0):
        total = 0
        for idx in range(len(self.density_plane)):
            plane_filtered = apply_gaussian_filter_2d(self.density_plane[idx], kernel_size, sigma)
            line_filtered = apply_gaussian_filter_1d(self.density_line[idx].squeeze(-1), kernel_size, sigma).unsqueeze(-1)
            total = total + F.mse_loss(self.density_plane[idx], plane_filtered) + F.mse_loss(self.density_line[idx], line_filtered)
        return total
    

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))
        return total
    

    def TV_loss_density(self, reg=TVLoss()):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2
        return total


    def update(self):
        reg = {
            "smoothness": self.smoothness(),
            "density_L1": self.density_L1(),
            "TV_loss_density": self.TV_loss_density() * 0.0,
        }

        return reg


    def forward(self, xyz_sampled, detach=True):
        ndim = xyz_sampled.ndim
        xyz_sampled_ = xyz_sampled.clone()
        xyz_sampled = xyz_sampled.view(-1,3)
        xyz_sampled = self.normalize_coord(xyz_sampled)

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)
        if detach:
            coordinate_plane = coordinate_plane.detach()
            coordinate_line = coordinate_line.detach()

        sigma_features = []

        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True, padding_mode="border").view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True, padding_mode="border").view(-1, *xyz_sampled.shape[:1])
            sigma_features.append((plane_coef_point * line_coef_point).T)

        sigma_features = torch.cat(sigma_features, dim=-1)

        if ndim == 3:
            sigma_features = sigma_features.view(xyz_sampled_.shape[0], xyz_sampled_.shape[1], -1)
        else:
            sigma_features = sigma_features.view(xyz_sampled_.shape[0], -1)

        return sigma_features


    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')


    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )

        t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
        correct_aabb = torch.zeros_like(new_aabb)
        correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
        correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
        print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
        new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))


