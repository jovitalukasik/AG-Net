import torch
import torch.nn as nn
import sys
from torch.autograd import Variable



def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

OPS = {
    'none' : lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3' : lambda C, stride, affine: AvgPoolBN(C, stride=stride),
    'max_pool_3x3' : lambda C, stride, affine: MaxPoolBN(C, stride=stride),
    'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
        ),
}


class AvgPoolBN(nn.Module):

    def __init__(self, C_out, stride):
        super(AvgPoolBN, self).__init__()
        self.op = nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.BatchNorm2d(C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        bn = self.op[1]
        bn, _ = BNWider(bn, new_C_out)
        self.op[1] = bn


class MaxPoolBN(nn.Module):

    def __init__(self, C_out, stride):
        super(MaxPoolBN, self).__init__()
        self.op = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.BatchNorm2d(C_out, affine=False)
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        bn = self.op[1]
        bn, _ = BNWider(bn, new_C_out)
        self.op[1] = bn


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        conv = self.op[1]
        bn = self.op[2]
        conv, _ = InChannelWider(conv, new_C_in)
        conv, index = OutChannelWider(conv, new_C_out)
        bn, _ = BNWider(bn, new_C_out, index=index)
        self.op[1] = conv
        self.op[2] = bn


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        conv1 = self.op[1]
        conv2 = self.op[2]
        bn = self.op[3]
        conv1, index = OutChannelWider(conv1, new_C_out)
        conv1.groups = new_C_in
        conv2, _ = InChannelWider(conv2, new_C_in, index=index)
        conv2, index = OutChannelWider(conv2, new_C_out)
        bn, _ = BNWider(bn, new_C_out, index=index)
        self.op[1] = conv1
        self.op[2] = conv2
        self.op[3] = bn


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)

    def wider(self, new_C_in, new_C_out):
        conv1 = self.op[1]
        conv2 = self.op[2]
        conv3 = self.op[5]
        conv4 = self.op[6]
        bn1 = self.op[3]
        bn2 = self.op[7]
        conv1, index = OutChannelWider(conv1, new_C_out)
        conv1.groups = new_C_in
        conv2, _ = InChannelWider(conv2, new_C_in, index=index)
        conv2, index = OutChannelWider(conv2, new_C_out)
        bn1, _ = BNWider(bn1, new_C_out, index=index)

        conv3, index = OutChannelWider(conv3, new_C_out)
        conv3.groups = new_C_in
        conv4, _ = InChannelWider(conv4, new_C_in, index=index)
        conv4, index = OutChannelWider(conv4, new_C_out)
        bn2, _ = BNWider(bn2, new_C_out, index=index)
        self.op[1] = conv1
        self.op[2] = conv2
        self.op[5] = conv3
        self.op[6] = conv4
        self.op[3] = bn1
        self.op[7] = bn2


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def wider(self, new_C_in, new_C_out):
        pass


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)

    def wider(self, new_C_in, new_C_out):
        pass


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

    def wider(self, new_C_in, new_C_out):
        self.conv_1, _ = InChannelWider(self.conv_1, new_C_in)
        self.conv_1, index1 = OutChannelWider(self.conv_1, new_C_out // 2)
        self.conv_2, _ = InChannelWider(self.conv_2, new_C_in)
        self.conv_2, index2 = OutChannelWider(self.conv_2, new_C_out // 2)
class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, depth = -1, use_stem=True):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3 if use_stem else min(3, C), C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if depth > 0 and i >= depth: break
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr
            if i == 2*layers//3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2*self._layers//3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if logits_aux is None: 
            return out, logits
        else: 
            return out, [logits, logits_aux]


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        # return logits, logits_aux
        if logits_aux is None: 
            return out, logits
        else: 
            return out, [logits, logits_aux]
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F


# bias = 0
def InChannelWider(module, new_channels, index=None):
    weight = module.weight
    in_channels = weight.size(1)

    if index is None:
        index = torch.randint(low=0, high=in_channels,
                              size=(new_channels-in_channels,))
    module.weight = nn.Parameter(
        torch.cat([weight, weight[:, index, :, :].clone()], dim=1), requires_grad=True)

    module.in_channels = new_channels
    module.weight.in_index = index
    module.weight.t = 'conv'
    if hasattr(weight, 'out_index'):
        module.weight.out_index = weight.out_index
    module.weight.raw_id = weight.raw_id if hasattr(
        weight, 'raw_id') else id(weight)
    return module, index


# bias = 0
def OutChannelWider(module, new_channels, index=None):
    weight = module.weight
    out_channels = weight.size(0)

    if index is None:
        index = torch.randint(low=0, high=out_channels,
                              size=(new_channels-out_channels,))
    module.weight = nn.Parameter(
        torch.cat([weight, weight[index, :, :, :].clone()], dim=0), requires_grad=True)

    module.out_channels = new_channels
    module.weight.out_index = index
    module.weight.t = 'conv'
    if hasattr(weight, 'in_index'):
        module.weight.in_index = weight.in_index
    module.weight.raw_id = weight.raw_id if hasattr(
        weight, 'raw_id') else id(weight)
    return module, index


def BNWider(module, new_features, index=None):
    running_mean = module.running_mean
    running_var = module.running_var
    if module.affine:
        weight = module.weight
        bias = module.bias
    num_features = module.num_features

    if index is None:
        index = torch.randint(low=0, high=num_features,
                              size=(new_features-num_features,))
    module.running_mean = torch.cat([running_mean, running_mean[index].clone()])
    module.running_var = torch.cat([running_var, running_var[index].clone()])
    if module.affine:
        module.weight = nn.Parameter(
            torch.cat([weight, weight[index].clone()], dim=0), requires_grad=True)
        module.bias = nn.Parameter(
            torch.cat([bias, bias[index].clone()], dim=0), requires_grad=True)
        
        module.weight.out_index = index
        module.bias.out_index = index
        module.weight.t = 'bn'
        module.bias.t = 'bn'
        module.weight.raw_id = weight.raw_id if hasattr(
            weight, 'raw_id') else id(weight)
        module.bias.raw_id = bias.raw_id if hasattr(
            bias, 'raw_id') else id(bias)
    module.num_features = new_features
    return module, index


def configure_optimizer(optimizer_old, optimizer_new):
    for i, p in enumerate(optimizer_new.param_groups[0]['params']):
        if not hasattr(p, 'raw_id'):
            optimizer_new.state[p] = optimizer_old.state[p]
            continue
        state_old = optimizer_old.state_dict()['state'][p.raw_id]
        state_new = optimizer_new.state[p]

        state_new['momentum_buffer'] = state_old['momentum_buffer']
        if p.t == 'bn':
            # BN layer
            state_new['momentum_buffer'] = torch.cat(
                [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id, p.out_index

        elif p.t == 'conv':
            # conv layer
            if hasattr(p, 'in_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][:, p.in_index, :, :].clone()], dim=1)
            if hasattr(p, 'out_index'):
                state_new['momentum_buffer'] = torch.cat(
                    [state_new['momentum_buffer'], state_new['momentum_buffer'][p.out_index, :, :, :].clone()], dim=0)
            # clean to enable multiple call
            del p.t, p.raw_id
            if hasattr(p, 'in_index'):
                del p.in_index
            if hasattr(p, 'out_index'):
                del p.out_index
    print('%d momemtum buffers loaded' % (i+1))
    return optimizer_new


def configure_scheduler(scheduler_old, scheduler_new):
    scheduler_new.load_state_dict(scheduler_old.state_dict())
    print('scheduler loaded')
    return scheduler_new
