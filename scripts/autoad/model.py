import torch
import torch.nn as nn
import torch.nn.functional as F


def _add_module(self, module):
    self.add_module(str(len(self) + 1), module)


nn.Sequential.add = _add_module


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun="LeakyReLU"):
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "Swish":
            return Swish()
        elif act_fun == "ELU":
            return nn.ELU()
        elif act_fun == "none":
            return nn.Sequential()
        else:
            raise ValueError(f"Unknown activation: {act_fun}")
    return act_fun()


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if all(s == min(inputs_shapes2) for s in inputs_shapes2) and all(
            s == min(inputs_shapes3) for s in inputs_shapes3
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 3
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=self.dim)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero"):
    if kernel_size > 1:
        to_pad = (kernel_size - 1) // 2
    else:
        to_pad = 0

    padder = None
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    if padder is not None:
        return nn.Sequential(padder, convolver)
    return convolver


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def skip(
    num_input_channels,
    num_output_channels,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=1,
    need_sigmoid=True,
    need_bias=True,
    pad="zero",
    upsample_mode="nearest",
    act_fun="LeakyReLU",
    need1x1_up=True,
):

    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)
    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_conn = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_conn, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(
                num_channels_skip[i]
                + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
            )
        )

        if num_channels_skip[i] != 0:
            skip_conn.add(
                conv(
                    input_depth,
                    num_channels_skip[i],
                    filter_skip_size,
                    bias=need_bias,
                    pad=pad,
                )
            )
            skip_conn.add(bn(num_channels_skip[i]))
            skip_conn.add(act(act_fun))

        deeper.add(
            conv(
                input_depth,
                num_channels_down[i],
                filter_size_down[i],
                2,
                bias=need_bias,
                pad=pad,
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(
            conv(
                num_channels_down[i],
                num_channels_down[i],
                filter_size_down[i],
                bias=need_bias,
                pad=pad,
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(
            conv(
                num_channels_skip[i] + k,
                num_channels_up[i],
                filter_size_up[i],
                1,
                bias=need_bias,
                pad=pad,
            )
        )
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(
                conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad)
            )
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


class AutoADNet(nn.Module):
    def __init__(
        self,
        num_channels,
        num_channels_down=None,
        num_channels_up=None,
        layers=5,
        lr=0.01,
        reg_noise_std=0.1,
    ):
        super().__init__()

        if num_channels_down is None:
            num_channels_down = [128] * layers
        if num_channels_up is None:
            num_channels_up = [128] * layers

        self.net = skip(
            num_input_channels=num_channels,
            num_output_channels=num_channels,
            num_channels_down=num_channels_down,
            num_channels_up=num_channels_up,
            num_channels_skip=num_channels_down,
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode="nearest",
            need_sigmoid=True,
            need_bias=True,
            pad="reflection",
            act_fun="LeakyReLU",
            need1x1_up=True,
        )

        self.reg_noise_std = reg_noise_std
        self.lr = lr

    def forward(self, x):
        return self.net(x)

    def get_regularization_noise(self, shape, device):
        if self.reg_noise_std > 0:
            return torch.randn(shape, device=device) * self.reg_noise_std
        return torch.zeros(shape, device=device)
