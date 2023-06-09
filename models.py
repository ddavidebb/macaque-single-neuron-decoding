"""
Deep neural networks adopted in 'Motor decoding from posterior parietal cortex using deep neural networks'.
by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso (submitted to Neurocomputing, 2022).

Author
------
Davide Borra, 2022
"""

import torch


class FCNN(torch.nn.Module):
    """
    Fully-connected neural network.

    Arguments
    ---------
    in_chans: int
        Number of recorded input neurons.
    n_classes: int
        Number of classes to decode.
    input_time_length: int
        Number of input time steps.
    n_layers: int
        Number of hidden fully-connected layers.
    n_units_per_layer : int
        Number of units in each hidden fully-connected layer.
    use_bn: int
        Flag to use Batch Normalization.
    drop_prob: float
        Dropout probability.
    act_fcn : str
        Hidden activation function (can be 'relu' or 'elu').
    """

    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 n_layers,
                 n_units_per_layer,
                 use_bn,
                 drop_prob,
                 act_fcn='elu'):
        super(FCNN, self).__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_time_length = input_time_length

        act = None
        if act_fcn == "relu":
            act = torch.nn.ReLU()
        elif act_fcn == "elu":
            act = torch.nn.ELU()

        self.classifier = torch.nn.Sequential()
        in_ = int(in_chans * input_time_length)

        for j in range(n_layers):
            self.classifier.add_module('fc_{0}'.format(j),
                                       torch.nn.Linear(in_, n_units_per_layer, bias=False if use_bn else True))
            if use_bn:
                self.classifier.add_module('bnorm_{0}'.format(j),
                                           torch.nn.BatchNorm1d(n_units_per_layer, momentum=0.01, affine=True,
                                                                eps=1e-3))
            self.classifier.add_module('act_{0}'.format(j), act)
            self.classifier.add_module('drop_{0}'.format(j), torch.nn.Dropout(p=drop_prob))
            in_ = n_units_per_layer

        self.classifier.add_module('fc_out',
                                   torch.nn.Linear(n_units_per_layer, n_classes, bias=True)
                                   )
        self.classifier.add_module('logsoftmax', torch.nn.LogSoftmax(dim=1))
        # initialization
        self.initialize_module(self.classifier)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        classifier_output = self.classifier(x)
        return classifier_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def initialize_module(self, module):
        for mod in module.modules():
            if hasattr(mod, 'weight'):
                if not ('BatchNorm' in mod.__class__.__name__):
                    torch.nn.init.xavier_uniform_(mod.weight, gain=1)
                else:
                    torch.nn.init.constant_(mod.weight, 1)
            if hasattr(mod, 'bias'):
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)


class CNN(torch.nn.Module):
    """
    Convolutional neural network defined by stacking convolutional blocks.
    In this class, 1d convolutions are used. These are formally equivalent to the 2d convolutions that were used, to ease CNNs description, in the associated publication.

    Arguments
    ---------
    in_chans: int
        Number of recorded input neurons.
    n_classes : int
        Number of classes to decode.
    input_time_length: int
        Number of input time steps.
    n_blocks : int
        Number of convolutional blocks.
    n_filter_conv : int
        Number of convolutional kernels learned in each convolutional layer.
    filter_size_conv : int
        Filter size along the temporal axis.
    use_bn : int
        Flag to use Batch Normalization.
    drop_prob : float
        Dropout probability.
    pool_fcn : str
        Pooling function (can be 'avg' or 'max').
    act_fcn : str
        Hidden activation function (can be 'relu' or 'elu').
    use_separable : int
        Flag to use separable convolutions in place of traditional convolutions.
    """

    def __init__(self,
                 in_chans,
                 n_classes,
                 input_time_length,
                 n_blocks,
                 n_conv_per_block,
                 n_filter_conv,
                 filter_size_conv,
                 use_bn,
                 drop_prob,
                 pool_fcn='avg',
                 act_fcn='elu',
                 use_separable=0,
                 add_output_act=True):
        super(CNN, self).__init__()
        pool = None
        act = None
        if pool_fcn == "avg":
            pool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        elif pool_fcn == "max":
            pool = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        if act_fcn == "relu":
            act = torch.nn.ReLU()
        elif act_fcn == "elu":
            act = torch.nn.ELU()

        self.conv_extractor = torch.nn.Sequential()
        in_ = in_chans
        for j in range(n_blocks):
            for k in range(n_conv_per_block):
                if not use_separable:
                    self.conv_extractor.add_module('conv_{0}_{1}'.format(j, k), torch.nn.Conv1d(in_,
                                                                                                n_filter_conv,
                                                                                                filter_size_conv,
                                                                                                stride=1,
                                                                                                bias=False if use_bn else True,
                                                                                                padding=filter_size_conv // 2)
                                                   )
                else:

                    self.conv_extractor.add_module('depth_conv_{0}_{1}'.format(j, k), torch.nn.Conv1d(in_,
                                                                                                      n_filter_conv,
                                                                                                      filter_size_conv,
                                                                                                      stride=1,
                                                                                                      bias=False,
                                                                                                      groups=in_,
                                                                                                      padding=filter_size_conv // 2)
                                                   )
                    self.conv_extractor.add_module('point_conv_{0}_{1}'.format(j, k),
                                                   torch.Conv1d(n_filter_conv, n_filter_conv,
                                                                1,
                                                                stride=1,
                                                                bias=False if use_bn else True,
                                                                padding=0))
                in_ = n_filter_conv
            if use_bn:
                self.conv_extractor.add_module('bnorm_{0}'.format(j), torch.nn.BatchNorm1d(n_filter_conv,
                                                                                           momentum=0.01, affine=True,
                                                                                           eps=1e-3))
            self.conv_extractor.add_module('act_{0}'.format(j), act)
            self.conv_extractor.add_module('pool_{0}'.format(j), pool)
            self.conv_extractor.add_module('drop_{0}'.format(j), torch.nn.Dropout(p=drop_prob))

        num_input_units_fc = (input_time_length // (n_blocks * 2)) * n_filter_conv
        self.classifier = torch.nn.Sequential()
        self.classifier.add_module('fc_0',
                                   torch.nn.Linear(num_input_units_fc, n_classes, bias=True)
                                   )
        if add_output_act:
            self.classifier.add_module('logsoftmax', torch.nn.LogSoftmax(dim=1))
        # initialization
        self.initialize_module(self.conv_extractor)
        self.initialize_module(self.classifier)

    def forward(self, x):
        conv_output = self.conv_extractor(x)
        classifier_input = conv_output.view(-1, self.num_flat_features(conv_output))
        classifier_output = self.classifier(classifier_input)
        return classifier_output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def initialize_module(self, module):
        for mod in module.modules():
            if hasattr(mod, 'weight'):
                if not ('BatchNorm' in mod.__class__.__name__):
                    torch.nn.init.xavier_uniform_(mod.weight, gain=1)
                else:
                    torch.nn.init.constant_(mod.weight, 1)
            if hasattr(mod, 'bias'):
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias, 0)


class RNN(torch.nn.Module):
    """
    Recurrent neural network defined by stacking Gated Recurrent Units (GRUs).

    Arguments
    ---------
    in_chans: int
        Number of recorded input neurons.
    n_classes : int
        Number of classes to decode.
    n_layers : int
        Number of stacked GRUs.
    n_hidden_feat_per_layer : int
        Number of features in the hidden state.
    drop_prob : float
        Dropout probability.
    """

    def __init__(self,
                 in_chans,
                 n_classes,
                 n_layers,
                 n_hidden_feat_per_layer,
                 drop_prob, ):

        super(RNN, self).__init__()

        self.rnn = torch.nn.Sequential()
        self.rnn.add_module('rnn', torch.nn.GRU(in_chans,
                                                n_hidden_feat_per_layer,
                                                num_layers=n_layers,
                                                dropout=drop_prob,
                                                batch_first=True))

        self.classifier = torch.nn.Sequential()
        self.classifier.add_module('dropout', torch.nn.Dropout(drop_prob))  # add dropout to last GRU stacked layer
        self.classifier.add_module('fc_out',
                                   torch.nn.Linear(n_hidden_feat_per_layer, n_classes, bias=True)
                                   )
        self.classifier.add_module('logsoftmax', torch.nn.LogSoftmax(dim=1))
        # initialization
        self.initialize_module()

    def forward(self, input):
        x = input.permute(0, 2, 1)
        s, hidden = self.rnn(x)
        s = self.classifier(s[:, -1, :])
        return s

    def initialize_module(self, ):
        for m in self.modules():
            if isinstance(m, (torch.nn.LSTM, torch.nn.GRU)):
                for param in m.parameters():
                    ih = (param.data for name,
                                         param in self.named_parameters() if 'weight_ih' in name)
                    hh = (param.data for name,
                                         param in self.named_parameters() if 'weight_hh' in name)
                    b = (param.data for name,
                                        param in self.named_parameters() if 'bias' in name)
                    for t in ih:
                        torch.nn.init.xavier_uniform_(t)
                    for t in hh:
                        torch.nn.init.orthogonal_(t)
                    for t in b:
                        t.fill_(0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # pdb.set_trace()
                m.bias.data.zero_()
