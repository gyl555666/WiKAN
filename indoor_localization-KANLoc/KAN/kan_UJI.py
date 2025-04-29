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

        self.output_layer = torch.nn.Linear(width[-1], 118).to(device)  # 确保输出层维度为118
        self.sigmoid = torch.nn.Sigmoid()

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
        x = self.output_layer(x)
        x = self.sigmoid(x)
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
import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

### EfficientKAN 代码导入
#from efficient_kan import KAN
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### 全局常量变量
# ------------------------------------------------------------------------
INPUT_DIM = 520
OUTPUT_DIM = 118
VERBOSE = 1

# 解析参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--gpu_id", help="运行此脚本的 GPU 设备 ID；默认是 0；设置为负数表示使用 CPU（即不使用 GPU）", default=0, type=int)
    parser.add_argument("-R", "--random_seed", help="随机种子", default=0, type=int)
    parser.add_argument("-E", "--epochs", help="训练轮数；默认是 20", default=200, type=int)
    parser.add_argument("-B", "--batch_size", help="批处理大小；默认是 10", default=10, type=int)
    parser.add_argument("-T", "--training_ratio", help="训练数据占总体数据的比例：默认是 0.90", default=0.9, type=float)
    parser.add_argument("-N", "--neighbours", help="定位时考虑的（最近）邻居位置数；默认是 1", default=8, type=int)
    parser.add_argument("--scaling", help="用于包含邻居位置的阈值比例（即阈值=比例*最大值）；默认是 0.0", default=0.2, type=float)
    args = parser.parse_args()

    # 使用命令行参数设置变量
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    #hidden_layers = [256, 128]  # 修改隐藏层神经元数
    N = args.neighbours
    scaling = args.scaling

    # 初始化随机种子生成器
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(random_seed)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    # 加载并预处理数据
    train_df = pd.read_csv('../data/UJIIndoorLoc/trainingData2.csv', header=0)
    train_AP_features = scale(np.asarray(train_df.iloc[:, 0:INPUT_DIM]).astype(float), axis=1)
    train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])),
                                          axis=1)

    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])
    for bld in blds:
        for flr in flrs:
            cond = (train_df['BUILDINGID'] == bld) & (train_df['FLOOR'] == flr)
            _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True)
            train_df.loc[cond, 'REFPOINT'] = idx

    blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
    train_labels = np.concatenate((blds, flrs, rfps), axis=1)

    train_val_split = np.random.rand(len(train_AP_features)) < training_ratio
    x_train = torch.tensor(train_AP_features[train_val_split], dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels[train_val_split], dtype=torch.float32).to(device)
    y_train = torch.clamp(y_train, 0, 1)  # 确保目标数据在 [0, 1] 范围内
    x_val = torch.tensor(train_AP_features[~train_val_split], dtype=torch.float32).to(device)
    y_val = torch.tensor(train_labels[~train_val_split], dtype=torch.float32).to(device)
    y_val = torch.clamp(y_val, 0, 1)  # 确保目标数据在 [0, 1] 范围内

    # 构建 KAN 模型
    model = KAN(width=[INPUT_DIM,  512,256, 118], grid=5, k=3, device='cuda')
    model.to(device)
    criterion = nn.BCELoss()
    weight_decay = 0.05
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=weight_decay)
    #optimizer = optim.LBFGS(model.parameters())
    '''optimizer = LBFGS(
        model.parameters(),
        lr=0.005,  # 你可以选择一个不同的学习率
        max_iter=100,  # 你可以增加最大迭代次数
        max_eval=150,  # 你可以增加最大函数评估次数
        tolerance_grad=1e-5,  # 你可以改变一阶最优性的容忍度
        tolerance_change=1e-8,  # 你可以改变函数值/参数变化的容忍度
        history_size=50,  # 你可以减少历史大小以减少内存使用
        line_search_fn='strong_wolfe'  # 你可以指定使用 strong Wolfe 线搜索
    )'''
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True)

    start_time = timer()

    # 训练循环
    for epoch in range(epochs):
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        model=model.to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        print(f'第 {epoch + 1}/{epochs} 轮, 损失: {loss.item()}, 验证损失: {val_loss.item()}')

        # 调整学习率
        scheduler.step(val_loss)



    # 加载测试数据并评估
    test_df = pd.read_csv('../data/UJIIndoorLoc/validationData2.csv', header=0)
    test_AP_features = scale(np.asarray(test_df.iloc[:, 0:INPUT_DIM]).astype(float), axis=1)
    x_test_utm = np.asarray(test_df['LONGITUDE'])
    y_test_utm = np.asarray(test_df['LATITUDE'])
    blds = np.asarray(pd.get_dummies(test_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(test_df['FLOOR']))
    test_labels = np.concatenate((blds, flrs), axis=1)

    x_test = torch.tensor(test_AP_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)
    y_test = torch.clamp(y_test, 0, 1)  # 确保目标数据在 [0, 1] 范围内

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')

    model.eval()
    with torch.no_grad():
        preds = model(x_test).cpu().numpy()

    blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    acc_bld = blds_results.mean()
    flrs_results = (np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    acc_flr = flrs_results.mean()
    acc_bf = (blds_results * flrs_results).mean()

    print(f'acc_bld: {acc_bld}')
    print(f'acc_flr: {acc_flr}')
    print(f'acc_bf: {acc_bf}')

    # 计算建筑和楼层正确估计时的定位误差
    mask = np.logical_and(blds_results, flrs_results)
    x_test_utm = x_test_utm[mask]
    y_test_utm = y_test_utm[mask]
    rfps = (preds[mask])[:, 8:118]

    n_success = len(x_test_utm)  # 使用实际的成功数量
    blds = blds[mask]
    flrs = flrs[mask]

    n_loc_failure = 0
    sum_pos_err = 0.0
    sum_pos_err_weighted = 0.0
    idxs = np.argpartition(rfps, -min(N, rfps.shape[1]), axis=1)[:, -min(N, rfps.shape[1]):]  # 确保 N 不超过 rfps 的大小
    threshold = scaling * np.amax(rfps, axis=1)

    for i in range(n_success):
        xs = []
        ys = []
        ws = []
        for j in idxs[i]:
            if j >= rfps.shape[1]:
                continue
            rfp = np.zeros(110)
            rfp[j] = 1
            rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1))
            if rows[0].size > 0:
                if rfps[i][j] >= threshold[i]:
                    xs.append(train_df.iloc[rows[0][0], 520])  # LONGITUDE
                    ys.append(train_df.iloc[rows[0][0], 521])  # LATITUDE
                    ws.append(rfps[i][j])
        if len(xs) > 0:
            sum_pos_err += math.sqrt((np.mean(xs) - x_test_utm[i]) ** 2 + (np.mean(ys) - y_test_utm[i]) ** 2)
            sum_pos_err_weighted += math.sqrt(
                (np.average(xs, weights=ws) - x_test_utm[i]) ** 2 + (np.average(ys, weights=ws) - y_test_utm[i]) ** 2)
        else:
            n_loc_failure += 1
    mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
    mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
    loc_failure = n_loc_failure / n_success

    print(f'mean_pos_err: {mean_pos_err}')
    print(f'mean_pos_err_weighted: {mean_pos_err_weighted}')
    # 将结果写入文件
    now = datetime.datetime.now()
    path_out = f'../results/xinkan_UJI{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out, 'w') as f:
        f.write("#+STARTUP: showall\n")  # unfold everything when opening
        f.write("* 系统参数\n")
        f.write(f"  - Numpy 随机数种子: {random_seed}\n")
        f.write(f"  - 训练数据占总体数据的比例: {training_ratio}\n")
        f.write(f"  - 训练轮数: {epochs}\n")
        f.write(f"  - 批处理大小: {batch_size}\n")
        f.write(f"  - 邻居数: {N}\n")
        f.write(f"  - 阈值比例: {scaling}\n")
        #f.write(f"  - 隐藏层结构: {hidden_layers}\n")
        f.write("* 性能\n")
        f.write(f"  - 建筑准确率: {acc_bld}\n")
        f.write(f"  - 楼层准确率: {acc_flr}\n")
        f.write(f"  - 建筑-楼层准确率: {acc_bf}\n")
        f.write(f"  - 定位失败率（给定正确的建筑/楼层）: {loc_failure}\n")
        f.write(f"  - 定位误差（米）: {mean_pos_err}\n")
        f.write(f"  - 加权定位误差（米）: {mean_pos_err_weighted}\n")

