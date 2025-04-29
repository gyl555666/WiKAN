import torch
from functools import reduce
from torch.optim import Optimizer

__all__ = ['LBFGS']


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        # print(f_prev, f_new, g_new)
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old low becomes new high
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # print(bracket)
    if len(bracket) == 1:
        t = bracket[0]
        f_new = bracket_f[0]
        g_new = bracket_g[0]
    else:
        t = bracket[low_pos]
        f_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            tolerance_ys=tolerance_ys,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        device = views[0].device
        return torch.cat(views, dim=0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """

        torch.manual_seed(0)

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        tolerance_ys = group['tolerance_ys']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > tolerance_ys:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss


import numpy as np
import torch
from sklearn.linear_model import LinearRegression
import sympy
import yaml
from sympy.utilities.lambdify import lambdify
import re

# sigmoid = sympy.Function('sigmoid')
# name: (torch implementation, sympy implementation)

# singularity protection functions
f_inv = lambda x, y_th: (
(x_th := 1 / y_th), y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th))
f_inv2 = lambda x, y_th: (
(x_th := 1 / y_th ** (1 / 2)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 2) * (torch.abs(x) >= x_th))
f_inv3 = lambda x, y_th: ((x_th := 1 / y_th ** (1 / 3)),
                          y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 3) * (
                                      torch.abs(x) >= x_th))
f_inv4 = lambda x, y_th: (
(x_th := 1 / y_th ** (1 / 4)), y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 4) * (torch.abs(x) >= x_th))
f_inv5 = lambda x, y_th: ((x_th := 1 / y_th ** (1 / 5)),
                          y_th / x_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x ** 5) * (
                                      torch.abs(x) >= x_th))
f_sqrt = lambda x, y_th: ((x_th := 1 / y_th ** 2), x_th / y_th * x * (torch.abs(x) < x_th) + torch.nan_to_num(
    torch.sqrt(torch.abs(x)) * torch.sign(x)) * (torch.abs(x) >= x_th))
f_power1d5 = lambda x, y_th: torch.abs(x) ** 1.5
f_invsqrt = lambda x, y_th: ((x_th := 1 / y_th ** 2),
                             y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (
                                         torch.abs(x) >= x_th))
f_log = lambda x, y_th: ((x_th := torch.e ** (-y_th)),
                         - y_th * (torch.abs(x) < x_th) + torch.nan_to_num(torch.log(torch.abs(x))) * (
                                     torch.abs(x) >= x_th))
f_tan = lambda x, y_th: ((clip := x % torch.pi), (delta := torch.pi / 2 - torch.arctan(y_th)),
                         - y_th / delta * (clip - torch.pi / 2) * (
                                     torch.abs(clip - torch.pi / 2) < delta) + torch.nan_to_num(torch.tan(clip)) * (
                                     torch.abs(clip - torch.pi / 2) >= delta))
f_arctanh = lambda x, y_th: ((delta := 1 - torch.tanh(y_th) + 1e-4),
                             y_th * torch.sign(x) * (torch.abs(x) > 1 - delta) + torch.nan_to_num(torch.arctanh(x)) * (
                                         torch.abs(x) <= 1 - delta))
f_arcsin = lambda x, y_th: (
(), torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1))
f_arccos = lambda x, y_th: (
(), torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1))
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(x) * (x <= x_th))

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
                'x^2': (lambda x: x ** 2, lambda x: x ** 2, 2, lambda x, y_th: ((), x ** 2)),
                'x^3': (lambda x: x ** 3, lambda x: x ** 3, 3, lambda x, y_th: ((), x ** 3)),
                'x^4': (lambda x: x ** 4, lambda x: x ** 4, 3, lambda x, y_th: ((), x ** 4)),
                'x^5': (lambda x: x ** 5, lambda x: x ** 5, 3, lambda x, y_th: ((), x ** 5)),
                '1/x': (lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
                '1/x^2': (lambda x: 1 / x ** 2, lambda x: 1 / x ** 2, 2, f_inv2),
                '1/x^3': (lambda x: 1 / x ** 3, lambda x: 1 / x ** 3, 3, f_inv3),
                '1/x^4': (lambda x: 1 / x ** 4, lambda x: 1 / x ** 4, 4, f_inv4),
                '1/x^5': (lambda x: 1 / x ** 5, lambda x: 1 / x ** 5, 5, f_inv5),
                'sqrt': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                'x^0.5': (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
                'x^1.5': (lambda x: torch.sqrt(x) ** 3, lambda x: sympy.sqrt(x) ** 3, 4, f_power1d5),
                '1/sqrt(x)': (lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
                '1/x^0.5': (lambda x: 1 / torch.sqrt(x), lambda x: 1 / sympy.sqrt(x), 2, f_invsqrt),
                'exp': (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
                'log': (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
                'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x), 3, lambda x, y_th: ((), torch.abs(x))),
                'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x), 2, lambda x, y_th: ((), torch.sin(x))),
                'cos': (lambda x: torch.cos(x), lambda x: sympy.cos(x), 2, lambda x, y_th: ((), torch.cos(x))),
                'tan': (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
                'tanh': (lambda x: torch.tanh(x), lambda x: sympy.tanh(x), 3, lambda x, y_th: ((), torch.tanh(x))),
                'sgn': (lambda x: torch.sign(x), lambda x: sympy.sign(x), 3, lambda x, y_th: ((), torch.sign(x))),
                'arcsin': (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
                'arccos': (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
                'arctan': (
                lambda x: torch.arctan(x), lambda x: sympy.atan(x), 4, lambda x, y_th: ((), torch.arctan(x))),
                'arctanh': (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
                '0': (lambda x: x * 0, lambda x: x * 0, 0, lambda x, y_th: ((), x * 0)),
                'gaussian': (lambda x: torch.exp(-x ** 2), lambda x: sympy.exp(-x ** 2), 3,
                             lambda x, y_th: ((), torch.exp(-x ** 2))),
                # 'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
                # 'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
                # 'relu': (lambda x: torch.relu(x), relu),
                }


def create_dataset(f,
                   n_var=2,
                   f_mode='col',
                   ranges=[-1, 1],
                   train_num=1000,
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset

    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.

    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']

    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:, i] = torch.rand(train_num, ) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        test_input[:, i] = torch.rand(test_num, ) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]

    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        print(f'f_mode {f_mode} not recognized')

    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)

    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset


def fit_params(x, y, fun, a_range=(-10, 10), b_range=(-10, 10), grid_number=101, iteration=3, verbose=True,
               device='cpu'):
    '''
    fit a, b, c, d such that

    .. math::
        |y-(cf(ax+b)+d)|^2

    is minimized. Both x and y are 1D array. Sweep a and b, find the best fitted model.

    Args:
    -----
        x : 1D array
            x values
        y : 1D array
            y values
        fun : function
            symbolic function
        a_range : tuple
            sweeping range of a
        b_range : tuple
            sweeping range of b
        grid_num : int
            number of steps along a and b
        iteration : int
            number of zooming in
        verbose : bool
            print extra information if True
        device : str
            device

    Returns:
    --------
        a_best : float
            best fitted a
        b_best : float
            best fitted b
        c_best : float
            best fitted c
        d_best : float
            best fitted d
        r2_best : float
            best r2 (coefficient of determination)

    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    '''
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing='ij')
        post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = torch.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0) ** 2
        denominator = torch.sum((post_fun - x_mean) ** 2, dim=0) * torch.sum((y - y_mean)[:, None, None] ** 2, dim=0)
        r2 = numerator / (denominator + 1e-4)
        r2 = torch.nan_to_num(r2)

        best_id = torch.argmax(r2)
        a_id, b_id = torch.div(best_id, grid_number, rounding_mode='floor'), best_id % grid_number

        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose == True:
                print('Best value at boundary.')
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]

        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]

    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(f'r2 is not very high, please double check if you are choosing the correct symbolic function.')

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(post_fun[:, None].detach().cpu().numpy(), y.detach().cpu().numpy())
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def sparse_mask(in_dim, out_dim):
    in_coord = torch.arange(in_dim) * 1 / in_dim + 1 / (2 * in_dim)
    out_coord = torch.arange(out_dim) * 1 / out_dim + 1 / (2 * out_dim)

    dist_mat = torch.abs(out_coord[:, None] - in_coord[None, :])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1, 0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1, 0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:, 0], all_connection[:, 1]] = 1.

    return mask


def add_symbolic(name, fun, c=1, fun_singularity=None):
    '''
    add a symbolic function to library

    Args:
    -----
        name : str
            name of the function
        fun : fun
            torch function or lambda function

    Returns:
    --------
        None

    Example
    -------
    >>> print(SYMBOLIC_LIB['Bessel'])
    KeyError: 'Bessel'
    >>> add_symbolic('Bessel', torch.special.bessel_j0)
    >>> print(SYMBOLIC_LIB['Bessel'])
    (<built-in function special_bessel_j0>, Bessel)
    '''
    exec(f"globals()['{name}'] = sympy.Function('{name}')")
    if fun_singularity == None:
        fun_singularity = fun
    SYMBOLIC_LIB[name] = (fun, globals()[name], c, fun_singularity)


def ex_round(ex1, n_digit):
    ex2 = ex1
    for a in sympy.preorder_traversal(ex1):
        if isinstance(a, sympy.Float):
            ex2 = ex2.subs(a, round(a, n_digit))
    return ex2


def augment_input(orig_vars, aux_vars, x):
    # if x is a tensor
    if isinstance(x, torch.Tensor):

        aux_values = torch.tensor([]).to(x.device)

        for aux_var in aux_vars:
            func = lambdify(orig_vars, aux_var, 'numpy')  # returns a numpy-ready function
            aux_value = torch.from_numpy(func(*[x[:, [i]].numpy() for i in range(len(orig_vars))]))
            aux_values = torch.cat([aux_values, aux_value], dim=1)

        x = torch.cat([aux_values, x], dim=1)

    # if x is a dataset
    elif isinstance(x, dict):
        x['train_input'] = augment_input(orig_vars, aux_vars, x['train_input'])
        x['test_input'] = augment_input(orig_vars, aux_vars, x['test_input'])

    return x


def batch_jacobian(func, x, create_graph=False):
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph)[0]


def batch_hessian(model, x, create_graph=False):
    # x in shape (Batch, Length)
    jac = lambda x: batch_jacobian(model, x, create_graph=True)

    def _jac_sum(x):
        return jac(x).sum(dim=0)

    return torch.autograd.functional.jacobian(_jac_sum, x, create_graph=create_graph).permute(1, 0, 2)


def create_dataset_from_data(inputs, labels, train_ratio=0.8, device='cpu'):
    num = inputs.shape[0]
    train_id = np.random.choice(num, int(num * train_ratio), replace=False)
    test_id = list(set(np.arange(num)) - set(train_id))
    dataset = {}
    dataset['train_input'] = inputs[train_id].detach().to(device)
    dataset['test_input'] = inputs[test_id].detach().to(device)
    dataset['train_label'] = labels[train_id].detach().to(device)
    dataset['test_label'] = labels[test_id].detach().to(device)

    return dataset


def get_derivative(model, inputs, labels, derivative='hessian', loss_mode='pred', reg_metric='w', lamb=0., lamb_l1=1.,
                   lamb_entropy=0.):
    def get_mapping(model):

        mapping = {}
        name = 'model1'

        keys = list(model.state_dict().keys())
        for key in keys:

            y = re.findall(".[0-9]+", key)
            if len(y) > 0:
                y = y[0][1:]
                x = re.split(".[0-9]+", key)
                mapping[key] = name + '.' + x[0] + '[' + y + ']' + x[1]

            y = re.findall("_[0-9]+", key)
            if len(y) > 0:
                y = y[0][1:]
                x = re.split(".[0-9]+", key)
                mapping[key] = name + '.' + x[0] + '[' + y + ']'

        return mapping

    # model1 = copy.deepcopy(model)
    model1 = model.copy()
    mapping = get_mapping(model)

    # collect keys and shapes
    keys = list(model.state_dict().keys())
    shapes = []

    for params in model.parameters():
        shapes.append(params.shape)

    # turn a flattened vector to model params
    def param2statedict(p, keys, shapes):

        new_state_dict = {}

        start = 0
        n_group = len(keys)
        for i in range(n_group):
            shape = shapes[i]
            n_params = torch.prod(torch.tensor(shape))
            new_state_dict[keys[i]] = p[start:start + n_params].reshape(shape)
            start += n_params

        return new_state_dict

    def differentiable_load_state_dict(mapping, state_dict, model1):

        for key in keys:
            if mapping[key][-1] != ']':
                exec(f"del {mapping[key]}")
            exec(f"{mapping[key]} = state_dict[key]")

    # input: p, output: output
    def get_param2loss_fun(inputs, labels):

        def param2loss_fun(p):

            p = p[0]
            state_dict = param2statedict(p, keys, shapes)
            # this step is non-differentiable
            # model.load_state_dict(state_dict)
            differentiable_load_state_dict(mapping, state_dict, model1)
            if loss_mode == 'pred':
                pred_loss = torch.mean((model1(inputs) - labels) ** 2, dim=(0, 1), keepdim=True)
                loss = pred_loss
            elif loss_mode == 'reg':
                reg_loss = model1.get_reg(reg_metric=reg_metric, lamb_l1=lamb_l1,
                                          lamb_entropy=lamb_entropy) * torch.ones(1, 1)
                loss = reg_loss
            elif loss_mode == 'all':
                pred_loss = torch.mean((model1(inputs) - labels) ** 2, dim=(0, 1), keepdim=True)
                reg_loss = model1.get_reg(reg_metric=reg_metric, lamb_l1=lamb_l1,
                                          lamb_entropy=lamb_entropy) * torch.ones(1, 1)
                loss = pred_loss + lamb * reg_loss
            return loss

        return param2loss_fun

    fun = get_param2loss_fun(inputs, labels)
    p = model2param(model)[None, :]
    if derivative == 'hessian':
        result = batch_hessian(fun, p)
    elif derivative == 'jacobian':
        result = batch_jacobian(fun, p)
    return result


def model2param(model):
    p = torch.tensor([]).to(model.device)
    for params in model.parameters():
        p = torch.cat([p, params.reshape(-1, )], dim=0)
    return p


import torch


def B_batch(x, grid, k=0, extend=True, device='cuda'):
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    '''# x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]'''

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]

    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value


def coef2curve(x_eval, grid, coef, k, device="cuda"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef

    b_splines = B_batch(x_eval, grid, k=k)  # (batch, in_dim, n_coef)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))

    return y_eval


def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    '''
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    '''
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]'''
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    # mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    mat = B_batch(x_eval, grid, k)  # (batch, in_dim, G+k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)  # (in_dim, out_dim, batch, n_coef)
    # coef shape: (in_dim, outdim, G+k)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)  # y_eval: (in_dim, out_dim, batch, 1)
    device = mat.device

    # coef = torch.linalg.lstsq(mat, y_eval,
    # driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]

    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n, n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:, :, :, 0]

    return coef


def extend_grid(grid, k_extend=0):
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


import torch
import torch.nn as nn
import numpy as np
import sympy


# from xkan.utils import *


class Symbolic_KANLayer(nn.Module):
    '''
    KANLayer class

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        funs: 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_name: 2D arry of str
            names of symbolic functions
        funs_sympy: 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine: 3D array of floats
            affine transformations of inputs and outputs

    Methods:
    --------
        __init__():
            initialize a Symbolic_KANLayer
        forward():
            forward
        get_subset():
            get subset of the KANLayer (used for pruning)
        fix_symbolic():
            fix an activation function to be symbolic
    '''

    def __init__(self, in_dim=3, out_dim=2, device='cpu'):
        '''
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)

        Args:
        -----
            in_dim : int
                input dimension
            out_dim : int
                output dimension
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=3)
        >>> len(sb.funs), len(sb.funs[0])
        (3, 3)
        '''
        super(Symbolic_KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim, device=device)).requires_grad_(False)
        # torch
        self.funs = [[lambda x: x * 0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        self.funs_avoid_singularity = [[lambda x, y_th: ((), x * 0.) for i in range(self.in_dim)] for j in
                                       range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: x * 0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?

        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
        # c*f(a*x+b)+d

        self.device = device
        self.to(device)

    def to(self, device):
        super(Symbolic_KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward

        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)

        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before summing on nodes

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        '''

        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j, i, 2] * \
                          self.funs_avoid_singularity[j][i](self.affine[j, i, 0] * x[:, [i]] + self.affine[j, i, 1],
                                                            torch.tensor(y_th))[1] + self.affine[j, i, 3]
                else:
                    xij = self.affine[j, i, 2] * self.funs[j][i](
                        self.affine[j, i, 0] * x[:, [i]] + self.affine[j, i, 1]) + self.affine[j, i, 3]
                postacts_.append(self.mask[j][i] * xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2, 1, 0, 3)[:, :, :, 0]
        y = torch.sum(postacts, dim=2)

        return y, postacts

    def get_subset(self, in_id, out_id):
        '''
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : Symbolic_KANLayer

        Example
        -------
        >>> sb_large = Symbolic_KANLayer(in_dim=10, out_dim=10)
        >>> sb_small = sb_large.get_subset([0,9],[1,2,3])
        >>> sb_small.in_dim, sb_small.out_dim
        (2, 3)
        '''
        sbb = Symbolic_KANLayer(self.in_dim, self.out_dim, device=self.device)
        sbb.in_dim = len(in_id)
        sbb.out_dim = len(out_id)
        sbb.mask.data = self.mask.data[out_id][:, in_id]
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:, in_id]
        return sbb

    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10, 10), b_range=(-10, 10),
                     verbose=True):
        '''
        fix an activation function to be symbolic

        Args:
        -----
            i : int
                the id of input neuron
            j : int
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True

        Returns:
        --------
            r2 (coefficient of determination)

        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)
        [['', '', ''], ['', '', 'sin']]
        Parameter containing:
        tensor([[0., 0., 0., 0.],
                 [0., 0., 0., 0.],
                 [1., 0., 1., 0.]], requires_grad=True)
        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        r2 is 0.9999701976776123
        [['', '', ''], ['', '', 'sin']]
        tensor([2.9981, 1.9997, 5.0039, 0.6978])
        '''
        if isinstance(fun_name, str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name

            if x == None or y == None:
                # initialzie from just fun
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                if random == False:
                    self.affine.data[j][i] = torch.tensor([1., 0., 1., 0.])
                else:
                    self.affine.data[j][i] = torch.rand(4, ) * 2 - 1
                return None
            else:
                # initialize from x & y and fun
                params, r2 = fit_params(x, y, fun, a_range=a_range, b_range=b_range, verbose=verbose,
                                        device=self.device)
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if random == False:
                self.affine.data[j][i] = torch.tensor([1., 0., 1., 0.])
            else:
                self.affine.data[j][i] = torch.rand(4, ) * 2 - 1
            return None

    def swap(self, i1, i2, mode='in'):

        with torch.no_grad():
            def swap_list_(data, i1, i2, mode='in'):

                if mode == 'in':
                    for j in range(self.out_dim):
                        data[j][i1], data[j][i2] = data[j][i2], data[j][i1]

                elif mode == 'out':
                    data[i1], data[i2] = data[i2], data[i1]

            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

                elif mode == 'out':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()

            swap_list_(self.funs_name, i1, i2, mode)
            swap_list_(self.funs_sympy, i1, i2, mode)
            swap_list_(self.funs_avoid_singularity, i1, i2, mode)
            swap_(self.affine.data, i1, i2, mode)
            swap_(self.mask.data, i1, i2, mode)


import torch
import torch.nn as nn
import numpy as np


# from xkan.spline import *
# from xkan.utils import sparse_mask


class KANLayer(nn.Module):
    """
    KANLayer class


    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device

    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5, scale_base=1.0, scale_base_mu=0.0,
                 scale_base_sigma=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, save_plot_data=True, device='cpu', sparse_init=False):
        ''''
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        (3, 5)
        '''
        super(KANLayer, self).__init__()
        # size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        ### grid size: (batch, in_dim, out_dim, G + 1) => (batch, in_dim, G + 2*k + 1)

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num + 1, self.in_dim, self.out_dim) - 1 / 2) * noise_scale / num
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))
        # if isinstance(scale_base, float):
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)

        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                                             scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) * 1 / np.sqrt(
            in_dim))
        # self.scale_base = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_base * mask).requires_grad_(sb_trainable)  # make scale trainable
        # else:
        # self.scale_base = torch.nn.Parameter(scale_base.to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * self.mask).requires_grad_(
            sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        self.grid_eps = grid_eps

        self.to(device)

    def to(self, device):
        super(KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        '''
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        '''
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        # x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        base = self.base_fun(x)  # (batch, in_dim)
        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)  # y shape: (batch, in_dim, out_dim)

        postspline = y.clone().permute(0, 2, 1)  # postspline shape: (batch, out_dim, in_dim)

        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * y
        y = self.mask[None, :, :] * y

        postacts = y.clone().permute(0, 2, 1)

        y = torch.sum(y, dim=1)  # shape (batch, out_dim)
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''

        batch = x.shape[0]
        # x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)

        self.grid.data = extend_grid(grid, k_extend=self.k)
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''

        batch = x.shape[0]

        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            h = (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]]) / num_interval
            grid_uniform = grid_adaptive[:, [0]] + h * torch.arange(num_interval + 1, )[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)

        grid = extend_grid(grid, k_extend=self.k)
        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def get_subset(self, in_id, out_id):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : KANLayer

        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun)
        spb.grid.data = self.grid[in_id]
        spb.coef.data = self.coef[in_id][:, out_id]
        spb.scale_base.data = self.scale_base[in_id][:, out_id]
        spb.scale_sp.data = self.scale_sp[in_id][:, out_id]
        spb.mask.data = self.mask[in_id][:, out_id]

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        return spb

    def swap(self, i1, i2, mode='in'):

        with torch.no_grad():
            def swap_(data, i1, i2, mode='in'):
                if mode == 'in':
                    data[i1], data[i2] = data[i2].clone(), data[i1].clone()
                elif mode == 'out':
                    data[:, i1], data[:, i2] = data[:, i2].clone(), data[:, i1].clone()

            if mode == 'in':
                swap_(self.grid.data, i1, i2, mode='in')
            swap_(self.coef.data, i1, i2, mode=mode)
            swap_(self.scale_base.data, i1, i2, mode=mode)
            swap_(self.scale_sp.data, i1, i2, mode=mode)
            swap_(self.mask.data, i1, i2, mode=mode)


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

# from .KAN  import * # Import the KAN model
import torch
import torch.nn as nn
import numpy as np
# from .KANLayer import *
# from .Symbolic_KANLayer import *
# from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
from sklearn.metrics import classification_report, precision_recall_fscore_support

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或者使用 ':16:8'
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

        self.output_layer_coords = nn.Linear(width[-1], 2).to(device)  # 输出经纬度坐标
        self.output_layer_floors = nn.Linear(width[-1], 5).to(device)  # 输出楼层分类

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
        coords_output = self.output_layer_coords(x)
        floors_output = self.output_layer_floors(x)
        return coords_output, floors_output

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


# Constants
INPUT_DIM = 992
OUTPUT_DIM = 5  # Adjust for floor classification
VERBOSE = 1

'''def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()'''

# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--gpu_id",
                        help="GPU device ID to run this script; default is 0; set to a negative number to use CPU",
                        default=0, type=int)
    parser.add_argument("-R", "--random_seed", help="Random seed", default=0, type=int)
    parser.add_argument("-E", "--epochs", help="Number of training epochs; default is 200", default=300, type=int)
    parser.add_argument("-B", "--batch_size", help="Batch size; default is 10", default=10, type=int)
    parser.add_argument("--dropout_prob", help="Dropout probability; default is 0.5", default=0.0, type=float)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    dropout_prob = args.dropout_prob

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(random_seed)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    # Load and preprocess data
    train_rss = pd.read_csv('../Tamperedata/Training_rss_21Aug17.csv', header=None)
    train_coords = pd.read_csv('../Tamperedata/Training_coordinates_21Aug17.csv', header=None)
    test_rss = pd.read_csv('../Tamperedata/Test_rss_21Aug17.csv', header=None)
    test_coords = pd.read_csv('../Tamperedata/Test_coordinates_21Aug17.csv', header=None)

    # Check shapes of loaded data
    print(f'Shape of train_rss: {train_rss.shape}')
    print(f'Shape of train_coords: {train_coords.shape}')
    print(f'Shape of test_rss: {test_rss.shape}')
    print(f'Shape of test_coords: {test_coords.shape}')

    # Ensure the data shapes are consistent
    assert train_rss.shape[0] == train_coords.shape[0], "train_rss和train_coords应该有相同的行数"
    assert test_rss.shape[0] == test_coords.shape[0], "test_rss和test_coords应该有相同的行数"

    train_rss = scale(train_rss, axis=1)
    test_rss = scale(test_rss, axis=1)

    # Extract floor information from coordinates (assuming floor is the third column)
    train_floors = train_coords.iloc[:, 2].astype(int)
    test_floors = test_coords.iloc[:, 2].astype(int)

    # Remap floor labels to be in range [0, OUTPUT_DIM-1]
    floor_mapping = {floor: idx for idx, floor in enumerate(np.unique(train_floors))}
    inverse_floor_mapping = {idx: floor for floor, idx in floor_mapping.items()}
    train_floors = train_floors.map(floor_mapping)
    test_floors = test_floors.map(floor_mapping)

    x_train = torch.tensor(train_rss, dtype=torch.float32).to(device)
    y_train_coords = torch.tensor(train_coords.iloc[:, :2].values, dtype=torch.float32).to(device)
    y_train_floors = torch.tensor(train_floors.values, dtype=torch.long).to(device)

    x_test = torch.tensor(test_rss, dtype=torch.float32).to(device)
    y_test_coords = torch.tensor(test_coords.iloc[:, :2].values, dtype=torch.float32).to(device)
    y_test_floors = torch.tensor(test_floors.values, dtype=torch.long).to(device)

    model = KAN(width=[INPUT_DIM, 128, 64, 5], grid=2, k=2,
                device='cuda')  # Adjust the width according to your requirement
    model.to(device)
    criterion_coords = nn.L1Loss()
    criterion_floors = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.00005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.95)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.95)

    train_losses = []
    val_losses = []
    start_time = timer()

    for epoch in range(epochs):
        model = model.to(device)
        x_train = x_train.to(device)
        y_train_coords = y_train_coords.to(device)
        y_train_floors = y_train_floors.to(device)

        model.train()
        optimizer.zero_grad()
        outputs_coords, outputs_floors = model(x_train)
        loss_coords = criterion_coords(outputs_coords, y_train_coords)
        loss_floors = criterion_floors(outputs_floors, y_train_floors)
        loss = loss_coords + loss_floors
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        '''model.eval()
        with torch.no_grad():
            val_outputs_coords, val_outputs_floors = model(x_test)
            val_loss_coords = criterion_coords(val_outputs_coords, y_test_coords)
            val_loss_floors = criterion_floors(val_outputs_floors, y_test_floors)
            val_loss = val_loss_coords + val_loss_floors'''

        train_losses.append(loss.item())
        # val_losses.append(val_loss.item())
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')

    # plot_loss_curve(train_losses, val_losses, 'xinkan_tam_training_validation_loss_curve.png')

    # Save model
    # torch.save(model.state_dict(), 'xinkan_tam_new_model.pth')
    model.load_state_dict(torch.load(f'../xkan/xinkan_tam_model1.pth'))
    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds_coords, preds_floors = model(x_test)

    # Calculate floor classification accuracy
    print(f'Shape of y_test_floors: {y_test_floors.shape}')
    print(f'Shape of preds_floors: {preds_floors.shape}')
    print(f'First few y_test_floors: {y_test_floors[:5]}')
    print(f'First few preds_floors: {preds_floors[:5]}')

    floor_results = (y_test_floors == torch.argmax(preds_floors, dim=1)).float()
    acc_flr = floor_results.mean().item()
    print(f'Floor accuracy: {acc_flr}')

    # Compute location error only in 2D (longitude and latitude) for correctly classified floors
    correct_indices = floor_results.nonzero(as_tuple=True)[0]
    pos_err = np.sqrt(
        np.sum((y_test_coords[correct_indices].cpu().numpy() - preds_coords[correct_indices].cpu().numpy()) ** 2,
               axis=1)).mean()
    print(f'Location error: {pos_err} meters')
    # Calculate classification metrics
    y_true = y_test_floors.cpu().numpy()
    y_pred = torch.argmax(preds_floors, dim=1).cpu().numpy()

    # 计算详细分类报告
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    # 打印指标
    print("\n=== Classification Metrics ===")
    print(f"macro Precision: {precision:.4f}")
    print(f"macro Recall: {recall:.4f}")
    print(f"macro F1 Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save results
    '''now = datetime.datetime.now()
    path_out = f'../results/xinkan_tam{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out, 'w') as f:
        f.write("#+STARTUP: showall\n")  # unfold everything when opening
        f.write("* System Parameters\n")
        f.write(f"  - Numpy Random Seed: {random_seed}\n")
        f.write(f"  - Number of Epochs: {epochs}\n")
        f.write(f"  - Batch Size: {batch_size}\n")
        f.write(f"  - Dropout Probability: {dropout_prob}\n")
        f.write(f"  - Hidden Layers: {[512, 256]}\n")
        f.write("* Performance\n")
        f.write(f"  - Floor Accuracy: {acc_flr}\n")
        f.write(f"  - Location Error (meters): {pos_err}\n")
    path_out1 = f'../results/xinkan_tam_CM{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out1, 'w') as f:
        f.write("* Classification Metrics\n")
        f.write(f"  - Macro Precision: {precision:.4f}\n")
        f.write(f"  - Macro Recall: {recall:.4f}\n")
        f.write(f"  - Macro F1 Score: {f1:.4f}\n")
        f.write("\n* Detailed Classification Report\n")
        f.write(classification_report(y_true, y_pred))'''