from collections import OrderedDict
import importlib.resources as importlib_resources
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_SCALER_REF = (
    importlib_resources.files('musmtl') / 'data' / 'sclr_dbmel3.dat.gz'
)


# TODO: this should be generalized at this moment
VALID_TASKS = {
    'self_', 'bpm', 'year', 'tag', 'taste', 'cdr_tag', 'lyrics', 'artist'
}
VALID_FUSION = {
    'null',  # MSS-CR
    '2',    # MS-CR@2
    '4',    # MS-CR@4
    '6',    # MS-CR@6
    'fc'    # MS-SR@FC
}


class VGGlikeMTL(nn.Module):
    """ VGG-like architecture for Multi-Task Learning
    """
    def __init__(self, tasks, branch_at, n_outs=50, n_ch_in=2):
        """
        Args:
            tasks (list of str): involved tasks
            branch_at (str): point where the network branches
                             ({'null, '2', '4', '6', 'fc'})
        """
        super(VGGlikeMTL, self).__init__()
        assert all([task.lower() in VALID_TASKS for task in tasks])
        assert branch_at in VALID_FUSION

        # build network
        self.n_outs = n_outs
        self.n_ch_in = n_ch_in
        self._build_net(tasks, branch_at)
        self.tasks = tasks
        self.branch_at = branch_at

    def _build_net(self, tasks, branch_at):
        """"""
        # build shared layers
        self.shared = nn.Sequential(*self._build_shared(branch_at))

        # list of layers (raw)
        self.branches_ = OrderedDict([
            (task, self._build_branch(task, branch_at))
            for task in tasks
        ])
        self._task2idx = {k:i for i, k in enumerate(self.branches_.keys())}

        # build feature extraction endpoint
        self.branches_feature = nn.ModuleList([
            nn.Sequential(*self.branches_[task][0])
            for task in tasks
        ])
        # build inference endpoint
        self.branches_infer = nn.ModuleList([
            nn.Sequential(*self.branches_[task][1])
            for task in tasks
        ])

    def _build_shared(self, branch_at):
        """"""
        if branch_at != 'null':
            shared = [ConvBlock2d(self.n_ch_in, 16, 5,
                                  conv_stride=(2, 1),
                                  pool_size=2)]
            if branch_at == '2':
                return shared  # (batch_sz, 16, 54, 64)

            elif branch_at == '4':
                shared.append(ConvBlock2d(16, 32, 3))
                shared.append(ConvBlock2d(32, 64, 3))
                return shared  # (batch_sz, 64, 13, 16)

            elif branch_at == '6':
                shared.append(ConvBlock2d(16, 32, 3))
                shared.append(ConvBlock2d(32, 64, 3))
                shared.append(ConvBlock2d(64, 64, 3))
                shared.append(ConvBlock2d(64, 128, 3))
                return shared  # (batch_sz, 128, 3, 4)

            elif branch_at == 'fc':
                shared.append(ConvBlock2d(16, 32, 3))
                shared.append(ConvBlock2d(32, 64, 3))
                shared.append(ConvBlock2d(64, 64, 3))
                shared.append(ConvBlock2d(64, 128, 3))
                shared.append(ConvBlock2d(128, 256, 3, pool_size=None))
                shared.append(ConvBlock2d(256, 256, 1, pool_size=None))
                shared.append(GlobalAveragePool())
                shared.append(linear_with_glorot_uniform(256, 256))
                shared.append(nn.BatchNorm1d(256))
                shared.append(nn.ReLU())
                shared.append(nn.Dropout())
                return shared  # (batch_sz, 256)
        else:
            return []  # no-sharing

    def _build_branch(self, task, branch_at):
        """"""
        # build feature part
        branch = []
        if branch_at != 'fc':
            if branch_at == 'null': # MSS-CR (null) : full branching
                branch.append(ConvBlock2d(self.n_ch_in, 16, 5,
                                          conv_stride=(2, 1),
                                          pool_size=2))
                branch.append(ConvBlock2d(16, 32, 3))
                branch.append(ConvBlock2d(32, 64, 3))
                branch.append(ConvBlock2d(64, 64, 3))
                branch.append(ConvBlock2d(64, 128, 3))

            elif branch_at == '2':
                branch.append(ConvBlock2d(16, 32, 3))
                branch.append(ConvBlock2d(32, 64, 3))
                branch.append(ConvBlock2d(64, 64, 3))
                branch.append(ConvBlock2d(64, 128, 3))

            elif branch_at == '4':
                branch.append(ConvBlock2d(64, 64, 3))
                branch.append(ConvBlock2d(64, 128, 3))

            branch.append(ConvBlock2d(128, 256, 3, pool_size=None))
            branch.append(ConvBlock2d(256, 256, 1, pool_size=None))
            branch.append(GlobalAveragePool())
            branch.append(linear_with_glorot_uniform(256, 256))
            branch.append(nn.BatchNorm1d(256))
            branch.append(nn.ReLU())
            branch.append(nn.Dropout())

        # build inference part
        branch_infer = []
        if task == 'self_':
            branch_infer.append(linear_with_glorot_uniform(512, 128))
            branch_infer.append(nn.BatchNorm1d(128))
            branch_infer.append(nn.ReLU())
            branch_infer.append(nn.Dropout())
            branch_infer.append(nn.Linear(128, 2))
        else:
            branch_infer.append(nn.Linear(256, self.n_outs))

        return branch, branch_infer

    def feature(self, X, task):
        """"""
        # TODO: check there's no explicit control flow for this
        # (for "fc" branch case)
        # (currently depends on the identity behavior of empty nn.ModuleList)
        X = self.shared(X)
        return self.branches_feature[self._task2idx[task]](X)

    def forward(self, X, task):
        """"""
        if task == 'self_':
            X_l = self.feature(X[0], task)
            X_r = self.feature(X[1], task)
            X = torch.cat([X_l, X_r], dim=-1)
        else:
            X = self.feature(X, task)

        return self.branches_infer[self._task2idx[task]](X)


class ConvBlock2d(nn.Module):
    """ Convolutional Building Block
    """
    def __init__(self, in_channel, out_channel, conv_kernel=3,
                 conv_stride=1, pool_size=2):
        """"""
        super(ConvBlock2d, self).__init__()
        self.pool_size = pool_size
        self.conv = nn.Conv2d(in_channel, out_channel,
                              conv_kernel, stride=conv_stride,
                              padding=conv_kernel // 2)
        torch.nn.init.xavier_uniform(self.conv.weight)  # Glorot uniform
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, X):
        """"""
        X = F.relu(self.bn(self.conv(X)))
        if self.pool_size is not None:
            X = F.max_pool2d(X, self.pool_size)
        return X


class GlobalAveragePool(nn.Module):
    """ Simple GAP layer for ease of Net building
    """
    def forward(self, X):
        return torch.mean(X.view(X.size(0), X.size(1), -1), dim=2)


class SpecStandardScaler(nn.Module):
    """ Standard Scaler for Temporal Axis on TF Representation
    """
    def __init__(self, mean, std, eps=1e-10):
        """
        Args:
            mean (np.ndarray): mean of spectrum
            std (np.ndarray): std of spectrum
        """
        super(SpecStandardScaler, self).__init__()
        self.mu = nn.Parameter(
            torch.from_numpy(mean)[None, None, None, :].float(),
            requires_grad=False
        )

        self.sigma = nn.Parameter(
            torch.max(
                torch.from_numpy(std).float(),
                torch.FloatTensor([eps])
            )[None, None, None, :],
            requires_grad=False
        )

    def forward(self, X):
        """"""
        return (X - self.mu) / self.sigma


def linear_with_glorot_uniform(f_in, f_out):
    """"""
    lin = nn.Linear(f_in, f_out)
    torch.nn.init.xavier_uniform(lin.weight)
    return lin
