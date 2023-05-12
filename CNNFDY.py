import torch.nn as nn
import torch


class FreqAttn(nn.Module):   # Version_40
    def __init__(self, freq_num):
        super(FreqAttn, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(freq_num, freq_num)

    def forward(self, x):
        lin = self.linear(x)
        freq_attn = self.sigmoid(lin.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(1))
        # print(freq_attn[0])
        res = x * freq_attn
        return res

# class FreqAttn(nn.Module):  # Version_41
#     def __init__(self, freq_num):
#         super(FreqAttn, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.conv = nn.Conv2d(freq_num, freq_num, (3, 1), stride=1, padding=1)
#
#     def forward(self, x):
#         after_conv = self.conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
#         freq_attn = self.sigmoid(after_conv.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(1))
#         # print(freq_attn[0])
#         res = x * freq_attn
#         return res


# class FreqAttn(nn.Module):   # Version_45
#     def __init__(self, freq_num):
#         super(FreqAttn, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(freq_num, freq_num)
#
#     def forward(self, x):
#         lin = self.linear(x)
#         mean_pool = lin.mean(dim=(1, 2), keepdim=True)
#         max_pool, _ = lin.max(dim=1, keepdim=True)
#         max_pool, _ = max_pool.max(dim=1, keepdim=True)
#         freq_attn = self.sigmoid(mean_pool + max_pool)
#         # print(freq_attn[0])
#         res = x * freq_attn
#         return res
class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
                                   requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x): #x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)    # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)    # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size) # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x): #x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  #x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  #x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  #x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)               #x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)               #x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)               #x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)               #x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)

class BottleNeck(nn.Module):
    """Residual block for ATST feature processing

    """
    def __init__(self, in_channels, out_channels):
        super(BottleNeck, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.res_block(x) + self.short_cut(x))


class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

# class ContextGating(nn.Module):
#     def __init__(self, input_num, freq_num):
#         super(ContextGating, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(input_num, input_num)
#         self.freq_attn = FreqAttn(freq_num)
#
#     def forward(self, x):
#         lin = self.linear(x.permute(0, 2, 3, 1))
#         lin = lin.permute(0, 3, 1, 2)
#         sig = self.sigmoid(lin)
#         res = x * sig
#         res = self.freq_attn(res)
#         return res


class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="cg",
        conv_dropout=0.5,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        atst_res_block=False,
        DY_layers=[0, 1, 1, 1, 1, 1, 1],
        temperature=31,
        n_basis_kernels=4,
        **transformer_kwargs
    ):
        """
            Initialization of CNN network s
        
        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        #print("Chek", nb_filters)
        #exit()
        cnn = nn.Sequential()
        freq_dim = [128, 64, 32, 16, 8, 4, 2, 1]
        self.atst_res_block = BottleNeck(24, 32) if atst_res_block else None

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            if i == 0:
                nIn = n_in_channel
            # input level
            # elif i == 2:
            #     if self.atst_res_block is not None:
            #         nIn = 64
            #     else:
            #         nIn = 56
            else:
                nIn = nb_filters[i - 1]
            nOut = nb_filters[i]
            if DY_layers[i] == 1:
                cnn.add_module("conv{0}".format(i), Dynamic_conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i],
                                                                   n_basis_kernels=n_basis_kernels,
                                                                   temperature=temperature, pool_dim='freq'))
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )

            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                # cnn.add_module("cg{0}".format(i), ContextGating(nOut, freq_num=freq_dim[i]))
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))
                # cnn.add_module("fa{0}".format(i), FreqAttn(freq_dim[i]))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        # 128x862x64
        # Front side of the CNN
        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels
        self.cnn = cnn

        # cnn = nn.Sequential()

        # for i in range(2, len(nb_filters)):
        #     conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
        #     cnn.add_module(
        #         "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
        #     )  # bs x tframe x mels
        # self.cnn_end = cnn

    def forward(self, r_spectrogram, v_spectrogram=None):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv front feature
        # r_front = self.cnn_front(r_spectrogram)

        # input level
        # # atst res block
        # if self.atst_res_block is not None:
        #     v_front = self.atst_res_block(v_spectrogram)
        # else:
        #     v_front = v_spectrogram
        # # conv feature fusion
        # x = torch.concat([r_front, v_front], dim=1)

        # # conv end feature
        # x = self.cnn_end(x)

        # hidden level
        x = self.cnn(r_spectrogram)
        return x
