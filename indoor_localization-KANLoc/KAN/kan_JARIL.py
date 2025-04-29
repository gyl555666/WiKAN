import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import os
import math
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import datetime
#from .KAN  import * # Import the KAN model
import torch
import torch.nn as nn
import numpy as np
from xkan.KANLayer import *
from xkan.Symbolic_KANLayer import *
from xkan.LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import matplotlib.pyplot as plt
import matplotlib
# 设置 Matplotlib 的默认字体为一个支持中文的字体，例如 'SimHei'（黑体）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.metrics import classification_report, precision_recall_fscore_support
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或者使用 ':16:8'
RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class KAN(nn.Module):
    '''
    KAN class

    Attributes:
    -----------
        biases: a list of nn.Linear()
            biases are added on nodes (in principle, biases can be absorbed into activation functions. However, we still have them for better optimization)
        act_fun: a list of KANLayer
            KANLayers
        depth: int
            depth of KAN
        width: list
            number of neurons in each layer. e.g., [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
        grid: int
            the number of grid intervals
        k: int
            the order of piecewise polynomial
        base_fun: fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun: a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled: bool
            If False, the symbolic front is not computed (to save time). Default: True.

    Methods:
    --------
        __init__():
            initialize a KAN
        initialize_from_another_model():
            initialize a KAN from another KAN (with the same shape, but potentially different grids)
        update_grid_from_samples():
            update spline grids based on samples
        initialize_grid_from_another_model():
            initalize KAN grids from another KAN
        forward():
            forward
        set_mode():
            set the mode of an activation function: 'n' for numeric, 's' for symbolic, 'ns' for combined (note they are visualized differently in plot(). 'n' as black, 's' as red, 'ns' as purple).
        fix_symbolic():
            fix an activation function to be symbolic
        suggest_symbolic():
            suggest the symbolic candicates of a numeric spline-based activation function
        lock():
            lock activation functions to share parameters
        unlock():
            unlock locked activations
        get_range():
            get the input and output ranges of an activation function
        plot():
            plot the diagram of KAN
        train():
            train KAN
        prune():
            prune KAN
        remove_edge():
            remove some edge of KAN
        remove_node():
            remove some node of KAN
        auto_symbolic():
            automatically fit all splines to be symbolic functions
        symbolic_formula():
            obtain the symbolic formula of the KAN network
    '''

    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, scale_base_mu=0.0, scale_base_sigma=1.0,
                 base_fun=torch.nn.SiLU(), symbolic_enabled=True, bias_trainable=False, grid_eps=1.0,
                 grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cuda', seed=0):
        '''
        initalize a KAN model

        Args:
        -----
            width : list of int
                :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            noise_scale : float
                initial injected noise to spline. Default: 0.1.
            base_fun : fun
                the residual function b(x). Default: torch.nn.SiLU().
            symbolic_enabled : bool
                compute or skip symbolic computations (for efficiency). By default: True.
            bias_trainable : bool
                bias parameters are updated or not. By default: True
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed

        Returns:
        --------
            self

        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> (model.act_fun[0].in_dim, model.act_fun[0].out_dim), (model.act_fun[1].in_dim, model.act_fun[1].out_dim)
        ((2, 5), (5, 1))
        '''
        super(KAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        self.biases = []
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width

        for l in range(self.depth):
            # splines
            # scale_base = 1 / np.sqrt(width[l]) + (torch.randn(width[l] * width[l + 1], ) * 2 - 1) * noise_scale_base
            scale_base = scale_base_mu * 1 / np.sqrt(width[l]) + \
                         scale_base_sigma * (torch.randn(width[l], width[l + 1], ) * 2 - 1) * 1 / np.sqrt(width[l])
            sp_batch = KANLayer(in_dim=width[l], out_dim=width[l + 1], num=grid, k=k, noise_scale=noise_scale,
                                scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps,
                                grid_range=grid_range, sp_trainable=sp_trainable,
                                sb_trainable=sb_trainable, device=device)
            self.act_fun.append(sp_batch)

            # bias
            bias = nn.Linear(width[l + 1], 1, bias=False, device=device).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        #self.output_layer = torch.nn.Linear(width[-1], 118).to(device)  # 确保输出层维度为118
        #self.sigmoid = torch.nn.Sigmoid()

        ### initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width[l], out_dim=width[l + 1], device=device)
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled

        self.device = device

    def initialize_from_another_model(self, another_model, x):
        '''
        initialize from a parent model. The parent has the same width as the current model but may have different grids.

        Args:
        -----
            another_model : KAN
                the parent model used to initialize the current model
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            self : KAN

        Example
        -------
        >>> model_coarse = KAN(width=[2,5,1], grid=5, k=3)
        >>> model_fine = KAN(width=[2,5,1], grid=10, k=3)
        >>> print(model_fine.act_fun[0].coef[0][0].data)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model_fine.initialize_from_another_model(model_coarse, x);
        >>> print(model_fine.act_fun[0].coef[0][0].data)
        tensor(-0.0030)
        tensor(0.0506)
        '''
        another_model(x.to(another_model.device))  # get activations
        batch = x.shape[0]

        self.initialize_grid_from_another_model(another_model, x.to(another_model.device))

        for l in range(self.depth):
            spb = self.act_fun[l]
            spb_parent = another_model.act_fun[l]

            # spb = spb_parent
            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            self.act_fun[l].coef.data = curve2coef(preacts.reshape(batch, spb.size).permute(1, 0),
                                                   postsplines.reshape(batch, spb.size).permute(1, 0), spb.grid,
                                                   k=spb.k, device=self.device)
            spb.scale_base.data = spb_parent.scale_base.data
            spb.scale_sp.data = spb_parent.scale_sp.data
            spb.mask.data = spb_parent.mask.data
            # print(spb.mask.data, self.act_fun[l].mask.data)

        for l in range(self.depth):
            self.biases[l].weight.data = another_model.biases[l].weight.data

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self

    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> print(model.act_fun[0].grid[0].data)
        >>> x = torch.rand(100,2)*5
        >>> model.update_grid_from_samples(x)
        >>> print(model.act_fun[0].grid[0].data)
        tensor([-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000])
        tensor([0.0128, 1.0064, 2.0000, 2.9937, 3.9873, 4.9809])
        '''
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def initialize_grid_from_another_model(self, model, x):
        '''
        initialize grid from a parent model

        Args:
        -----
            model : KAN
                parent model
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model_parent = KAN(width=[1,1], grid=5, k=3)
        >>> model_parent.act_fun[0].grid.data = torch.linspace(-2,2,steps=6)[None,:]
        >>> x = torch.linspace(-2,2,steps=1001)[:,None]
        >>> model = KAN(width=[1,1], grid=5, k=3)
        >>> print(model.act_fun[0].grid.data)
        >>> model = model.initialize_from_another_model(model_parent, x)
        >>> print(model.act_fun[0].grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-2.0000, -1.2000, -0.4000,  0.4000,  1.2000,  2.0000]])
        '''
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

    def forward(self, x):
        '''
        KAN forward

        Args:
        -----
            x : 2D torch.float
                inputs, shape (batch, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (batch, output dimension)

        Example
        -------
        >>> model = KAN(width=[2,5,3], grid=5, k=3)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x).shape
        torch.Size([100, 3])
        '''

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            x_symbolic = 0.
            postacts_symbolic = 0.

            x = x_numerical + x_symbolic
            postacts = postacts_numerical + postacts_symbolic

            # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
            # grid_reshape = self.act_fun[l].grid.reshape(self.width[l + 1], self.width[l], -1)
            # input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            input_range = torch.std(preacts, dim=0) + 0.1
            output_range = torch.std(postacts, dim=0)
            self.acts_scale.append(output_range / input_range)
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)
        #x = self.output_layer(x)
        #x = self.sigmoid(x)
        return x

    def set_mode(self, l, i, j, mode, mask_n=None):
        '''
        set (l,i,j) activation to have mode

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            mode : str
                'n' (numeric) or 's' (symbolic) or 'ns' (combined)
            mask_n : None or float)
                magnitude of the numeric front

        Returns:
        --------
            None
        '''
        if mode == "s":
            mask_n = 0.;
            mask_s = 1.
        elif mode == "n":
            mask_n = 1.;
            mask_s = 0.
        elif mode == "sn" or mode == "ns":
            if mask_n == None:
                mask_n = 1.
            else:
                mask_n = mask_n
            mask_s = 1.
        else:
            mask_n = 0.;
            mask_s = 0.

        self.act_fun[l].mask.data[j * self.act_fun[l].in_dim + i] = mask_n
        self.symbolic_fun[l].mask.data[j, i] = mask_s

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True,
                     random=False):
        '''
        set (l,i,j) activation to be symbolic (specified by fun_name)

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]

        Returns:
        --------
            None or r2 (coefficient of determination)

        Example 1
        ---------
        >>> # when fit_params_bool = False
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=False)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])

        Example 2
        ---------
        >>> # when fit_params_bool = True
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # obtain activations (otherwise model does not have attributes acts)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=True)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        r2 is 0.8131332993507385
        r2 is not very high, please double check if you are choosing the correct symbolic function.
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])
        '''
        self.set_mode(l, i, j, mode="s")
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            return None
        else:
            x = self.acts[l][:, i]
            y = self.spline_postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range,
                                                   verbose=verbose)
            return r2

    def unfix_symbolic(self, l, i, j):
        '''
        unfix the (l,i,j) activation function.
        '''
        self.set_mode(l, i, j, mode="n")

    def unfix_symbolic_all(self):
        '''
        unfix all activation functions.
        '''
        for l in range(len(self.width) - 1):
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    self.unfix_symbolic(l, i, j)

    def lock(self, l, ids):
        '''
        lock ids in the l-th layer to be the same function

        Args:
        -----
            l : int
                layer index
            ids : 2D list
                :math:`[[i_1,j_1],[i_2,j_2],...]` set :math:`(l,i_i,j_1), (l,i_2,j_2), ...` to be the same function

        Returns:
        --------
            None

        Example
        -------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        >>> model.lock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        tensor([[0, 1],
                [2, 3],
                [4, 5]])
        tensor([[0, 1],
                [2, 1],
                [4, 5]])
        '''
        self.act_fun[l].lock(ids)

    def unlock(self, l, ids):
        '''
        unlock ids in the l-th layer to be the same function

        Args:
        -----
            l : int
                layer index
            ids : 2D list)
                [[i1,j1],[i2,j2],...] set (l,ii,j1), (l,i2,j2), ... to be unlocked

        Example:
        --------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> model.lock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        >>> model.unlock(0,[[1,0],[1,1]])
        >>> print(model.act_fun[0].weight_sharing.reshape(3,2))
        tensor([[0, 1],
                [2, 1],
                [4, 5]])
        tensor([[0, 1],
                [2, 3],
                [4, 5]])
        '''
        self.act_fun[l].unlock(ids)

    def get_range(self, l, i, j, verbose=True):
        '''
        Get the input range and output range of the (l,i,j) activation

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index

        Returns:
        --------
            x_min : float
                minimum of input
            x_max : float
                maximum of input
            y_min : float
                minimum of output
            y_max : float
                maximum of output

        Example
        -------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # do a forward pass to obtain model.acts
        >>> model.get_range(0,0,0)
        x range: [-2.13 , 2.75 ]
        y range: [-0.50 , 1.83 ]
        (tensor(-2.1288), tensor(2.7498), tensor(-0.5042), tensor(1.8275))
        '''
        x = self.spline_preacts[l][:, j, i]
        y = self.spline_postacts[l][:, j, i]
        x_min = torch.min(x)
        x_max = torch.max(x)
        y_min = torch.min(y)
        y_max = torch.max(y)
        if verbose:
            print('x range: [' + '%.2f' % x_min, ',', '%.2f' % x_max, ']')
            print('y range: [' + '%.2f' % y_min, ',', '%.2f' % y_max, ']')
        return x_min, x_max, y_min, y_max

    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False,
             in_vars=None, out_vars=None, title=None):
        '''
        plot KAN

        Args:
        -----
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title

        Returns:
        --------
            Figure

        Example
        -------
        >>> # see more interactive examples in demos
        >>> model = KAN(width=[2,3,1], grid=3, k=3, noise_scale=1.0)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # do a forward pass to obtain model.acts
        >>> model.plot()
        '''
        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width[l]):
                for j in range(self.width[l + 1]):
                    rank = torch.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    symbol_mask = self.symbolic_fun[l].mask[j][i]
                    numerical_mask = self.act_fun[l].mask.reshape(self.width[l + 1], self.width[l])[j][i]
                    if symbol_mask > 0. and numerical_mask > 0.:
                        color = 'purple'
                        alpha_mask = 1
                    if symbol_mask > 0. and numerical_mask == 0.:
                        color = "red"
                        alpha_mask = 1
                    if symbol_mask == 0. and numerical_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if symbol_mask == 0. and numerical_mask == 0.:
                        color = "white"
                        alpha_mask = 0

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    plt.plot(self.acts[l][:, i][rank].cpu().detach().numpy(),
                             self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                    if sample == True:
                        plt.scatter(self.acts[l][:, i][rank].cpu().detach().numpy(),
                                    self.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color,
                                    s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    lock_id = self.act_fun[l].lock_id[j * self.width[l] + i].long().item()
                    if lock_id > 0:
                        # im = plt.imread(f'{folder}/lock.png')
                        im = plt.imread(f'{RESOURCE_DIR}/lock.png')
                        newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                        plt.text(500, 400, lock_id, fontsize=15)
                        newax.imshow(im)
                        newax.axis('off')

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        alpha = [score2alpha(score.cpu().detach().numpy()) for score in self.acts_scale]

        # draw skeleton
        width = np.array(self.width)
        A = 1
        y0 = 0.4  # 0.4

        # plt.figure(figsize=(5,5*(neuron_depth-1)*y0))
        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width), 5)

        max_neuron = np.max(width)
        max_num_weights = np.max(width[:-1] * width[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 3)

        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0))
        # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

        # plot scatters and lines
        for l in range(neuron_depth):
            n = width[l]
            spacing = A / n
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

                if l < neuron_depth - 1:
                    # plot connections
                    n_next = width[l + 1]
                    N = n * n_next
                    for j in range(n_next):
                        id_ = i * n_next + j

                        symbol_mask = self.symbolic_fun[l].mask[j][i]
                        numerical_mask = self.act_fun[l].mask.reshape(self.width[l + 1], self.width[l])[j][i]
                        if symbol_mask == 1. and numerical_mask == 1.:
                            color = 'purple'
                            alpha_mask = 1.
                        if symbol_mask == 1. and numerical_mask == 0.:
                            color = "red"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 1.:
                            color = "black"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 0.:
                            color = "white"
                            alpha_mask = 0.
                        if mask == True:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1],
                                     color=color, lw=2 * scale,
                                     alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                     [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale,
                                     alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                        else:
                            plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1],
                                     color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                            plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                     [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale,
                                     alpha=alpha[l][j][i] * alpha_mask)

            plt.xlim(0, 1)
            plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        plt.axis('off')

        # plot splines
        for l in range(neuron_depth - 1):
            n = width[l]
            for i in range(n):
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                    up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                    if mask == False:
                        newax.imshow(im, alpha=alpha[l][j][i])
                    else:
                        ### make sure to run model.prune() first to compute mask ###
                        newax.imshow(im, alpha=alpha[l][j][i] * self.mask[l][i].item() * self.mask[l + 1][j].item())
                    newax.axis('off')

        if in_vars != None:
            n = self.width[0]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=40 * scale,
                                             horizontalalignment='center', verticalalignment='center')

        if out_vars != None:
            n = self.width[-1]
            for i in range(n):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), y0 * (len(self.width) - 1) + 0.1, out_vars[i],
                                             fontsize=40 * scale, horizontalalignment='center',
                                             verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, y0 * (len(self.width) - 1) + 0.2, title, fontsize=40 * scale,
                                         horizontalalignment='center', verticalalignment='center')
def get_label_to_coord_mapping(num_labels=16):
    """
    将位置标签映射到二维坐标。
    根据用户指定的逻辑，将16个标签映射到4x4网格上，每个网格间距0.8米。

    Args:
        num_labels (int): 位置标签的数量。

    Returns:
        list of tuples: 每个标签对应的 (x, y) 坐标。
    """
    if num_labels != 16:
        raise ValueError("当前映射逻辑仅支持16个标签。请根据需要调整映射函数。")
    coords = np.array([
        [i // 4* 0.8, i % 4 * 0.8] for i in range(16)
    ])
    return coords.tolist()


# 计算平均定位误差 (ALE)
def compute_average_error(predictions, targets, label_to_coord):
    """
    计算平均定位误差 (ALE)，单位为米。

    Args:
        predictions (list or np.array): 预测的标签列表。
        targets (list or np.array): 真实的标签列表。
        label_to_coord (list of tuples): 标签到二维坐标的映射列表。

    Returns:
        float: 平均定位误差 (米)。
    """
    pred_coords = [label_to_coord[pred] for pred in predictions]
    true_coords = [label_to_coord[true] for true in targets]
    pred_coords = np.array(pred_coords)
    true_coords = np.array(true_coords)
    errors = np.linalg.norm(pred_coords - true_coords, axis=1)
    average_error = errors.mean()
    return average_error
def main():
    # 设置参数
    #batch_size=256
    batch_size = 256
    num_epochs = 600
    #learning_rate = 0.05
    learning_rate = 0.004
    #milestone_steps = [10,50, 100, 150]
    milestone_steps = [10,50,100,150]#学习率调整的里程碑
    #gamma = 0.5  # 学习率衰减因子
    gamma = 0.5  # 学习率衰减因子
    timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建结果保存目录
    os.makedirs('kan_weights', exist_ok=True)
    os.makedirs('kan_result', exist_ok=True)
    os.makedirs('kan_vis', exist_ok=True)

    # 加载训练数据
    print("Loading training data...")
    train_data_amp = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_data']
    train_location_label = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_location_label']
    train_labels_combined = train_location_label.squeeze()
    num_train_instances = train_data_amp.shape[0]

    train_data = torch.from_numpy(train_data_amp).float()
    train_labels = torch.from_numpy(train_labels_combined).long()

    train_dataset = TensorDataset(train_data, train_labels)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据
    print("Loading test data...")
    test_data_amp = sio.loadmat('../csi_data/test_data_split_amp.mat')['test_data']
    test_location_label = sio.loadmat('../csi_data/test_data_split_amp.mat')['test_location_label']
    test_labels_combined = test_location_label.squeeze()
    num_test_instances = test_data_amp.shape[0]

    test_data = torch.from_numpy(test_data_amp).float()
    test_labels = torch.from_numpy(test_labels_combined).long()

    test_dataset = TensorDataset(test_data, test_labels)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 定义位置标签到二维坐标的映射
    num_labels = max(train_labels.max().item(), test_labels.max().item()) + 1
    if num_labels != 16:
        raise ValueError(f"当前标签数量为 {num_labels}，但映射函数仅支持16个标签。请根据需要调整映射函数。")
    label_to_coord = get_label_to_coord_mapping(num_labels=num_labels)

    train_data = torch.mean(train_data, dim=-1)[0]
    test_data = torch.mean(test_data, dim=-1)[0]
    #train_data = torch.mean(train_data, dim=-1)
    #test_data = torch.mean(test_data, dim=-1)
    # 初始化模型
    print("Initializing KAN model...")
    #layers_hidden = [52, 512, 256,  num_labels]  # 隐藏层配置，请根据需要调整
    kan_model = KAN(width=[52,  512,256, 16],grid=5,
                    k=3).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_steps, gamma=gamma)

    # 训练和评估记录
    train_loss_loc = np.zeros(num_epochs)
    test_loss_loc = np.zeros(num_epochs)
    train_acc_loc = np.zeros(num_epochs)
    test_acc_loc = np.zeros(num_epochs)
    test_avg_error = np.zeros(num_epochs)

    best_test_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_model_path = ""  # 初始化保存路径
    # 开始训练
    start_time=timer()
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        kan_model.train()


        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for samples, labels in tqdm(train_data_loader, desc='Training'):
            # 确保输入数据为 [batch_size, 52]
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)  # 对时间维度进行池化
            samples = samples.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = kan_model(samples)
            loss = criterion(outputs, labels)
            #total_loss = loss

            loss.backward()
            #total_loss.backward()
            optimizer.step()

            running_loss += loss.item() * samples.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        scheduler.step()
        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train
        train_loss_loc[epoch] = epoch_loss
        train_acc_loc[epoch] = epoch_acc

        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f'Training completed in {elapsed_time:.2f} seconds')
        # 测试阶段
        kan_model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        all_test_predictions = []
        all_test_labels = []

        with torch.no_grad():
            for samples, labels in tqdm(test_data_loader, desc='Testing'):
                if len(samples.shape) > 2:
                    samples = samples.mean(dim=-1)  # 对时间维度进行池化
                samples = samples.cuda()
                labels = labels.cuda()

                outputs = kan_model(samples)
                loss = criterion(outputs, labels)

                running_loss_test += loss.item() * samples.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

                all_test_predictions.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        epoch_loss_test = running_loss_test / total_test
        epoch_acc_test = 100.0 * correct_test / total_test
        avg_error_test = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)

        test_loss_loc[epoch] = epoch_loss_test
        test_acc_loc[epoch] = epoch_acc_test
        test_avg_error[epoch] = avg_error_test

        print(
            f'Test Loss: {epoch_loss_test:.4f}, Test Accuracy: {epoch_acc_test:.2f}%, Test Avg Error: {avg_error_test:.4f}m')
        print("\nClassification Report for Epoch {}:".format(epoch + 1))
        print(classification_report(all_test_labels, all_test_predictions, digits=4, zero_division=0))

        # 计算并打印宏平均
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_test_labels, all_test_predictions, average='macro', zero_division=0
        )

        print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        # 保存最佳模型
        if epoch_acc_test > best_test_acc:
            best_test_acc = epoch_acc_test
            best_precision = precision_macro
            best_recall = recall_macro
            best_f1 = f1_macro
            best_model_path = f'kan_weights/kan_model_best_epoch{epoch + 1}_Acc{epoch_acc_test:.3f}.pth'  # 添加路径记录
            torch.save(kan_model.state_dict(), best_model_path)  # 取消注释
            print(f'Best model saved at epoch {epoch + 1} with Test Accuracy: {epoch_acc_test:.3f}%')
            #torch.save(kan_model.state_dict(), f'kan_weights/kan_model_best_epoch{epoch + 1}_Acc{epoch_acc_test:.3f}.pth')
            #print(f'Best model saved at epoch {epoch + 1} with Test Accuracy: {epoch_acc_test:.3f}%')

    # 保存训练和测试结果
    print("Saving results...")
    sio.savemat(f'kan_result/train_loss_loc_{timestamp}.mat', {'train_loss_loc': train_loss_loc})
    sio.savemat(f'kan_result/test_loss_loc_{timestamp}.mat', {'test_loss_loc': test_loss_loc})
    sio.savemat(f'kan_result/train_acc_loc_{timestamp}.mat', {'train_acc_loc': train_acc_loc})
    sio.savemat(f'kan_result/test_acc_loc_{timestamp}.mat', {'test_acc_loc': test_acc_loc})
    sio.savemat(f'kan_result/test_avg_error_{timestamp}.mat', {'test_avg_error': test_avg_error})
    sio.savemat(f'kan_result/best_precision_{timestamp}.mat', {'best_precision': best_precision})
    sio.savemat(f'kan_result/best_recall_{timestamp}.mat', {'best_recall': best_recall})
    sio.savemat(f'kan_result/best_f1_{timestamp}.mat', {'best_f1': best_f1})
    print("Training and testing completed.")
    # 加载最佳模型
    kan_model.load_state_dict(torch.load(best_model_path))
    kan_model.eval()

    # 重新运行测试集预测
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for samples, labels in test_data_loader:
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)
            samples = samples.cuda()
            labels = labels.cuda()

            outputs = kan_model(samples)
            _, predicted = torch.max(outputs.data, 1)

            all_test_predictions.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # 计算最终指标
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    final_acc = 100.0 * np.mean(np.array(all_test_predictions) == np.array(all_test_labels))

    print(f"\n最佳模型最终测试准确率: {final_acc:.2f}%")
    print(f"最佳模型最终平均定位误差: {final_avg_error:.4f}m")
    # 最终评估
    print("\nFinal Evaluation on Test Set:")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall : {best_recall:.4f}")
    print(f"F1 Score : {best_f1:.4f}")

    # 计算最终的平均定位误差
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    print(f"Final Average Localization Error: {final_avg_error:.4f}m")
    sio.savemat(f'kan_result/final_avg_error_{timestamp}.mat', {'final_avg_error': final_avg_error})
    # 保存最终预测结果
    sio.savemat(f'kan_vis/locResult_{timestamp}.mat', {'loc_prediction': np.array(all_test_predictions)})

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_loc, label='Training loss', linewidth=2)
    plt.plot(test_loss_loc, label='Testing loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.title(f'训练和验证损失曲线（最佳验证准确率：{best_test_acc:.2f}%）', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # 保存高清图片
    loss_curve_path = f'kan_vis/loss_curve_{timestamp}.png'
    plt.savefig(loss_curve_path, dpi=500, bbox_inches='tight')
    print(f'\n损失曲线已保存至：{loss_curve_path}')

    plt.show()


if __name__ == "__main__":
    main()
