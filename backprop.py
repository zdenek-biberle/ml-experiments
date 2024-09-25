# Description: Attempting to implement basic backpropagation in Python

import math
import typing
import numpy.typing as npt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
import inspect


class Num:
    def __init__(self, value: float):
        self.value = value
        self.grad = 0
        self.num_deps = 0

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __sub__(self, other):
        return Difference(self, other)

    def relu(self):
        return ReLU(self)

    def leaky_relu(self, alpha):
        return LeakyReLU(self, alpha)

    def backwards(self):
        assert self.num_deps == 0, "backwards() must be called on the result"
        self.num_deps = 1
        self.add_grad(1)

    # Add a gradient to this number. Backpropagate if all gradients are in.
    def add_grad(self, grad):
        self.grad += grad
        self.num_deps -= 1
        if self.num_deps == 0:
            self.backprop()

    # Backpropagate the gradient to the parents.
    # Should be implemented by subclasses.
    def backprop(self):
        pass

    def __str__(self) -> str:
        return f"Num(value={self.value}, grad={self.grad}, num_deps={self.num_deps})"


class Parameter(Num):
    def __init__(self, value: float, name: str):
        super().__init__(value)
        self.name = name

    def __str__(self) -> str:
        return f"Parameter(name={self.name}, value={self.value}, grad={self.grad}, num_deps={self.num_deps})"


class Sum(Num):
    def __init__(self, a: Num, b: Num):
        super().__init__(a.value + b.value)
        self.a = a
        self.b = b
        a.num_deps += 1
        b.num_deps += 1

    def backprop(self):
        self.a.add_grad(self.grad)
        self.b.add_grad(self.grad)


class Difference(Num):
    def __init__(self, a, b):
        super().__init__(a.value - b.value)
        self.a = a
        self.b = b
        a.num_deps += 1
        b.num_deps += 1

    def backprop(self):
        self.a.add_grad(self.grad)
        self.b.add_grad(-self.grad)


class Product(Num):
    def __init__(self, a, b):
        super().__init__(a.value * b.value)
        self.a = a
        self.b = b
        a.num_deps += 1
        b.num_deps += 1

    def backprop(self):
        self.a.add_grad(self.grad * self.b.value)
        self.b.add_grad(self.grad * self.a.value)


class ReLU(Num):
    def __init__(self, x):
        super().__init__(relu(x.value))
        self.x = x
        x.num_deps += 1

    def backprop(self):
        self.x.add_grad(self.grad if self.x.value > 0 else 0.)


class LeakyReLU(Num):
    def __init__(self, x, alpha):
        super().__init__(leaky_relu(x.value, alpha))
        self.x = x
        self.alpha = alpha
        x.num_deps += 1

    def backprop(self):
        self.x.add_grad(self.grad if self.x.value >
                        0 else self.alpha * self.grad)


def relu(x: float | npt.NDArray | Num) -> float | npt.NDArray | Num:
    if isinstance(x, Num):
        return x.relu()
    elif isinstance(x, np.ndarray):
        return np.maximum(0., x)
    else:
        return max(0., x)


def leaky_relu(x: float | npt.NDArray | Num, alpha: float) -> float | npt.NDArray | Num:
    if isinstance(x, Num):
        return x.leaky_relu(alpha)
    elif isinstance(x, np.ndarray):
        return np.maximum(alpha * x, x)
    else:
        return max(alpha * x, x)


def sanity_check():
    a = Num(2.)
    b = Num(3.)
    c = Num(4.)
    d = a * b + c
    d.backwards()
    assert a.grad == 3
    assert b.grad == 2
    assert c.grad == 1
    a = Num(2.)
    b = a * a
    c = b * b
    d = c * c
    d.backwards()
    assert a.grad == 1024
    a = Num(2.)
    b = a.relu()
    b.backwards()
    assert a.grad == 1
    a = Num(-2.)
    b = a.relu()
    b.backwards()
    assert a.grad == 0
    a = Num(2.)
    b = a.leaky_relu(.2)
    b.backwards()
    assert a.grad == 1
    a = Num(-2.)
    b = a.leaky_relu(.2)
    b.backwards()
    assert a.grad == .2


sanity_check()


# An optimizer that we'll use to train the model
class GradientDescent:
    def __init__(self, parameters: list[Parameter], learning_rate: float, momentum: float = 0.4):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.last_grads = None
        self.momentum = momentum

    def step(self):
        grads = np.zeros(len(self.parameters))
        for idx, parameter in enumerate(self.parameters):
            assert parameter.num_deps == 0
            grads[idx] = parameter.grad
        max_grad = np.max(np.abs(grads))

        # This is a bit of a hack to prevent the gradients from exploding.
        if max_grad > 100:
            print(f"Warning: max grad is {max_grad}")
            grads = grads / (max_grad / 100)

        # apply momentum
        if self.last_grads is None:
            grads_with_momentum = grads
        else:
            grads_with_momentum = grads * (1 - self.momentum) + self.last_grads * self.momentum
        self.last_grads = grads

        for idx, parameter in enumerate(self.parameters):
            parameter.value -= grads_with_momentum[idx] * self.learning_rate
            parameter.grad = 0


# This is the big boy - this thing will try to approximate the given training data
# using the given model. It will then plot a bunch of stuff.
def approximate(
    # takes an input value + some number of parameters and returns a result, must work with both Num and ndarray
    model: typing.Callable,
    training_xs: list[float],
    training_ys: list[float],
    representative_xs: npt.NDArray,
    representative_ys: npt.NDArray,
    learning_rate=0.05,
    num_epochs=250,
    batch_size=100,
):
    # establish random initial parameters of the model
    parameters = [Parameter(random.uniform(-5, 5), name)
                  for name in inspect.getfullargspec(model).args[1:]]      
    num_parameters = len(parameters) 
    print(f"Initial parameters: { {p.name: p.value for p in parameters} }")

    # define the loss function
    def loss(parameters, x, y):
        diff = model(x, *parameters) - y
        return diff * diff

    # define the optimizer
    optimizer = GradientDescent(parameters, learning_rate)

    # here we'll store the history of the training process for the purpose of plotting
    # we skip the first epoch 'cause it usually sucks
    history_len = num_epochs - 1
    history_parameter_values: npt.NDArray = np.zeros(
        (num_parameters, history_len))
    history_parameter_grads: npt.NDArray = np.zeros(
        (num_parameters, history_len))
    history_loss = np.zeros(history_len)

    def plot_range(nums, extend=0.25) -> tuple[float, float]:
        nums_min = np.min(nums)
        nums_max = np.max(nums)
        diff = nums_max - nums_min
        nums_min -= extend * diff
        nums_max += extend * diff
        return nums_min, nums_max

    plt.ion()
    figure = plt.figure()
    xlim = plot_range(representative_xs)
    ylim = plot_range(representative_ys)
    ax = figure.add_subplot(1, 2, 1, xlabel='x',
                            ylabel='y', xlim=xlim, ylim=ylim)
    ax.plot(representative_xs, representative_ys)
    model_plot_xs = np.linspace(*xlim, 100)
    model_plot, = ax.plot([], [], color='red')
    ax_loss_progress = figure.add_subplot(
        1, 2, 2, xlabel='epoch', ylabel='loss', yscale='log')
    ax_loss_progress.autoscale(True)
    loss_progress_plot, = ax_loss_progress.plot([1])

    # training loop
    for epoch in range(num_epochs):
        # shuffle indices of the training data so that we can train in random batches
        indices = list(range(len(training_xs)))
        random.shuffle(indices)

        epoch_loss = 0
        for batch_data_index in range(0, len(training_xs), batch_size):

            batch_data_indices = indices[batch_data_index:batch_data_index + batch_size]
            batch_losses = []
            for index in batch_data_indices:
                x = training_xs[index]
                y = training_ys[index]
                l = loss(parameters, Num(float(x)), Num(float(y)))
                if math.isnan(l.value):
                    print(f"NaN loss at epoch {epoch}, index {index}, x {x}, y {
                          y}, parameters {[p.value for p in parameters]}")
                    raise ValueError("NaN loss")

                batch_losses.append(l)

            batch_loss = sum(batch_losses, start=Num(0.)) * \
                Num(1 / len(batch_data_indices))
            if batch_data_index == 0:
                for param_idx, parameter in enumerate(parameters):
                    history_parameter_values[param_idx,
                                             epoch - 1] = parameter.value
                    history_parameter_grads[param_idx,
                                            epoch - 1] = parameter.grad
                    history_loss[epoch - 1] = batch_loss.value

            batch_loss.backwards()
            epoch_loss += batch_loss.value
            optimizer.step()

        if epoch % 50 == 0 and epoch > 0:
            print(f"Epoch {epoch}, loss {epoch_loss}")
            parameter_values = [p.value for p in parameters]
            model_plot.set_data(model_plot_xs, model(
                model_plot_xs, *parameter_values))
            loss_progress_plot.set_data(np.arange(epoch), history_loss[:epoch])
            ax_loss_progress.relim()
            ax_loss_progress.autoscale_view()
            figure.canvas.draw()
            figure.canvas.flush_events()

    grid_spec = figure.add_gridspec(2, 2, width_ratios=[1, 2])

    ax.set_subplotspec(grid_spec[0, 0])
    ax_loss_progress.set_subplotspec(grid_spec[1, 0])

    ax3d: axes3d.Axes3D = figure.add_subplot(
        grid_spec[0, 1], projection='3d', xlabel='idk', ylabel='idk', zlabel='loss', zscale='log')
    parameter_loss_wireframe = None
    parameter_loss_plot, = ax3d.plot([1], [1], [1], color='red', zorder=3)

    ax3d_epoch: axes3d.Axes3D = figure.add_subplot(
        grid_spec[1, 1], projection='3d', xlabel='epoch', ylabel='idk', zlabel='loss', zscale='log')
    parameter_epoch_loss, = ax3d_epoch.plot(
        [1], [1], [1], color='red', zorder=3)
    parameter_epoch_loss_wireframe = None

    loss_surface_precision = 32

    def update_line_plot(frame: int):
        model_plot.set_data(model_plot_xs, model(
            model_plot_xs, *history_parameter_values[:, frame]))
        return [model_plot]

    def update_parameter_plot(frame: tuple[int, int]):
        a_idx, b_idx = frame
        a_values = history_parameter_values[a_idx]
        b_values = history_parameter_values[b_idx]
        a_min, a_max = plot_range(a_values)
        b_min, b_max = plot_range(b_values)
        a_space = np.linspace(a_min, a_max, loss_surface_precision)
        b_space = np.linspace(b_min, b_max, loss_surface_precision)
        a_space, b_space = np.meshgrid(a_space, b_space)
        all_parameter_values: list[float | npt.NDArray] = [
            p.value for p in parameters]
        all_parameter_values[a_idx] = a_space[..., np.newaxis]
        all_parameter_values[b_idx] = b_space[..., np.newaxis]
        loss_values = np.mean(
            loss(all_parameter_values, representative_xs, representative_ys), axis=-1)
        nonlocal parameter_loss_wireframe
        if parameter_loss_wireframe:
            parameter_loss_wireframe.remove()
        parameter_loss_wireframe = ax3d.plot_surface(
            a_space, b_space, loss_values, zorder=2, color='tab:blue')  # , cmap=cm.viridis)
        parameter_loss_plot.set_data_3d(
            history_parameter_values[a_idx], history_parameter_values[b_idx], history_loss)
        ax3d.set_xlabel(parameters[a_idx].name)
        ax3d.set_ylabel(parameters[b_idx].name)
        ax3d.set_xlim(a_min, a_max)
        ax3d.set_ylim(b_min, b_max)
        ax3d.set_zlim(np.min(loss_values), np.max(loss_values))
        return [parameter_loss_wireframe, parameter_loss_plot]

    def update_parameter_epoch_loss_plot(param_idx: int):
        epoch_values = np.arange(history_len)
        epoch_lim = (0, history_len - 1)
        param_values = history_parameter_values[param_idx]
        param_lim = plot_range(param_values, extend=1)
        loss_values = history_loss
        loss_lim = plot_range(loss_values)
        parameter_epoch_loss.set_data_3d(
            epoch_values, param_values, loss_values)
        ax3d_epoch.set_ylabel(parameters[param_idx].name)
        ax3d_epoch.set_xlim(epoch_lim)
        ax3d_epoch.set_ylim(param_lim)
        ax3d_epoch.set_zlim(loss_lim)
        nonlocal parameter_epoch_loss_wireframe
        if parameter_epoch_loss_wireframe:
            parameter_epoch_loss_wireframe.remove()

        # we want to draw the evolution of the loss function with respect to the parameter,
        # i.e. a surface where one dimension is a variation in the parameter, one dimension is time (epoch)
        # and the third dimension is the loss value.
        # We obviously do not know the loss at all those points in space, so we'll have to evaluate it here,
        # however we'll evaluate against the representative data, so at least it's not that much to compute.
        # We also don't want to evaluate the loss at every single epoch, so we'll downsample in the epoch
        # dimension a bit.
        # The hardest part here is getting all the array dimensions just right.

        # P: width of the parameter space
        # E: width of the epoch space (i.e. downsampled history_len)
        # R: width of the representative data
        # param_space:   P
        # epoch_space:   E
        # meshgridded param space: P x E
        # meshgridded epoch space: P x E

        # other params:  1 x E x 1 (i.e. they need to vary in time)
        # this param:    P x 1 x 1 (i.e. it needs to vary in parameter space)
        # repr xs:               R (i.e. they need to vary in the representative data)
        # repr ys:               R (same here)
        # loss(p, x, y): P x E x R
        # mean loss:     P x E

        surface_epoch_stride = history_len // loss_surface_precision
        param_space = np.linspace(*param_lim, loss_surface_precision * 2)
        epoch_space = np.arange(0, history_len, surface_epoch_stride)
        all_parameters: list[npt.NDArray] = [
            p[np.newaxis, ::surface_epoch_stride, np.newaxis] for p in history_parameter_values]
        this_param = param_space[..., np.newaxis, np.newaxis]
        all_parameters[param_idx] = this_param
        loss_values = np.mean(
            loss(all_parameters, representative_xs, representative_ys), axis=-1)
        epoch_space, param_space = np.meshgrid(epoch_space, param_space)
        parameter_epoch_loss_wireframe = ax3d_epoch.plot_surface(
            epoch_space, param_space, loss_values, zorder=2, color='tab:blue')  # , cmap=cm.viridis)

        return [parameter_epoch_loss, parameter_epoch_loss_wireframe]

    results_animation = animation.FuncAnimation(
        figure, update_line_plot, frames=history_len, interval=50)
    parameter_frames = [(a, b) for a in range(num_parameters)
                        for b in range(a + 1, num_parameters)]
    parameter_animation = animation.FuncAnimation(
        figure, update_parameter_plot, frames=parameter_frames, interval=2500)
    parameter_epoch_loss_animation = animation.FuncAnimation(
        figure, update_parameter_epoch_loss_plot, frames=num_parameters, interval=2500)
    plt.ioff()
    plt.show()


# and now some examples


def approximate_line():
    def model(x, a, b):
        return a * x + b

    ideal_a = 2.5
    ideal_b = -1.234
    training_xs = [random.random() for _ in range(100)]
    training_ys = [model(x, ideal_a, ideal_b) +
                   random.uniform(-0.1, 0.1) for x in training_xs]
    representative_xs = np.linspace(0, 1, 100)
    representative_ys = model(representative_xs, ideal_a, ideal_b)

    approximate(
        model=model,
        training_xs=training_xs,
        training_ys=training_ys,
        representative_xs=representative_xs,
        representative_ys=representative_ys,
        batch_size=20,
        num_epochs=100,
    )


def approximate_relu():
    def model(x, a, b):
        return relu(a * x + b)

    ideal_a = 2.5
    ideal_b = -1.234
    training_xs = [random.uniform(-1, 1) for _ in range(1000)]
    training_ys = [model(x, ideal_a, ideal_b) +
                   random.uniform(-0.1, 0.1) for x in training_xs]
    representative_xs = np.linspace(0, 1, 100)
    representative_ys = model(representative_xs, ideal_a, ideal_b)

    approximate(
        model=model,
        training_xs=training_xs,
        training_ys=training_ys,
        representative_xs=representative_xs,
        representative_ys=representative_ys,
    )


def approximate_relu_with_bias():
    def data(x, a, b, c):
        return relu(a * x + b) + c

    def model(x, a, b, c):
        return relu(x * a + b) + c

    a = 2.
    b = 1.5
    c = 1.3
    training_xs = [random.uniform(-4, 3) for _ in range(500)]
    training_ys = [data(x, a, b, c) + random.uniform(-0.1, 0.1)
                   for x in training_xs]
    representative_xs = np.linspace(-4, 3, 100)
    representative_ys = model(representative_xs, a, b, c)

    approximate(
        model=model,
        training_xs=training_xs,
        training_ys=training_ys,
        representative_xs=representative_xs,
        representative_ys=representative_ys,
    )


def approximate_sin_with_1_hidden_layer():
    def data(x):
        return np.sin(x * .5 * np.pi)

    def model(
        x,
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
        d0
    ):
        return leaky_relu(x * a0 + b0, .2) * c0 + \
            leaky_relu(x * a1 + b1, .2) * c1 + \
            leaky_relu(x * a2 + b2, .2) * c2 + \
            leaky_relu(x * a3 + b3, .2) * c3 + \
            leaky_relu(x * a4 + b4, .2) * c4 + \
            leaky_relu(x * a5 + b5, .2) * c5 + \
            leaky_relu(x * a6 + b6, .2) * c6 + \
            leaky_relu(x * a7 + b7, .2) * c7 + \
            leaky_relu(x * a8 + b8, .2) * c8 + \
            leaky_relu(x * a9 + b9, .2) * c9 + \
            leaky_relu(x * a10 + b10, .2) * c10 + \
            leaky_relu(x * a11 + b11, .2) * c11 + \
            leaky_relu(x * a12 + b12, .2) * c12 + \
            leaky_relu(x * a13 + b13, .2) * c13 + \
            leaky_relu(x * a14 + b14, .2) * c14 + \
            leaky_relu(x * a15 + b15, .2) * c15 + \
            d0

    training_xs = np.array([random.uniform(-5, 5)
                           for _ in range(1000)], dtype=np.float64)
    training_ys = data(training_xs)
    representative_xs = np.linspace(-5, 5, 100)
    representative_ys = data(representative_xs)

    approximate(
        model=model,
        training_xs=training_xs,
        training_ys=training_ys,
        representative_xs=representative_xs,
        representative_ys=representative_ys,
        num_epochs=1000,
        batch_size=100,
        learning_rate=0.01,
    )


def approximate_sin_with_2_hidden_layers():
    def data(x):
        return np.sin(x * .5 * np.pi)

    def model(
        x,
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5, b6,
        c11, c12, c13, c14, c15, c16,
        c21, c22, c23, c24, c25, c26,
        c31, c32, c33, c34, c35, c36,
        c41, c42, c43, c44, c45, c46,
        c51, c52, c53, c54, c55, c56,
        c61, c62, c63, c64, c65, c66,
        d1, d2, d3, d4, d5, d6,
        e1, e2, e3, e4, e5, e6,
        f0
    ):
        # first hidden layer
        h11 = leaky_relu(x * a1 + b1, .1)
        h12 = leaky_relu(x * a2 + b2, .1)
        h13 = leaky_relu(x * a3 + b3, .1)
        h14 = leaky_relu(x * a4 + b4, .1)
        h15 = leaky_relu(x * a5 + b5, .1)
        h16 = leaky_relu(x * a6 + b6, .1)

        # second hidden layer
        h21 = leaky_relu(h11 * c11 + h12 * c12 + h13 * c13 + h14 * c14 + h15 * c15 + h16 * c16 + d1, .1)
        h22 = leaky_relu(h11 * c21 + h12 * c22 + h13 * c23 + h14 * c24 + h15 * c25 + h16 * c26 + d2, .1)
        h23 = leaky_relu(h11 * c31 + h12 * c32 + h13 * c33 + h14 * c34 + h15 * c35 + h16 * c36 + d3, .1)
        h24 = leaky_relu(h11 * c41 + h12 * c42 + h13 * c43 + h14 * c44 + h15 * c45 + h16 * c46 + d4, .1)
        h25 = leaky_relu(h11 * c51 + h12 * c52 + h13 * c53 + h14 * c54 + h15 * c55 + h16 * c56 + d5, .1)
        h26 = leaky_relu(h11 * c61 + h12 * c62 + h13 * c63 + h14 * c64 + h15 * c65 + h16 * c66 + d6, .1)

        return h21 * e1 + h22 * e2 + h23 * e3 + h24 * e4 + h25 * e5 + h26 * e6 + f0

    training_xs = np.array([random.uniform(-5, 5)
                           for _ in range(1000)], dtype=np.float64)
    training_ys = data(training_xs)
    representative_xs = np.linspace(-5, 5, 100)
    representative_ys = data(representative_xs)

    approximate(
        model=model,
        training_xs=training_xs,
        training_ys=training_ys,
        representative_xs=representative_xs,
        representative_ys=representative_ys,
        num_epochs=1000,
        batch_size=30,
        learning_rate=0.001,
    )


approximate_sin_with_2_hidden_layers()
