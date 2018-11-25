from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
from tensorboardX import SummaryWriter

try:
    import chainer
    import chainer.functions as F
    import chainer.links as L
    chainer_installed = True
except ImportError:
    print('Chainer is not installed, skipping test')
    chainer_installed = False


if chainer_installed:
    class ChainerGraphTest(unittest.TestCase):
        def test_MLP(self):
            class MLP(chainer.Chain):
                def __init__(self):
                    super(MLP, self).__init__()
                    n_hidden_unit = 32
                    n_class = 3

                    with self.init_scope():
                        self.linear_0 = L.Linear(n_hidden_unit)
                        self.linear_1 = L.Linear(n_class)

                def forward(self, x):
                    h = self.linear_0(x)
                    h = F.relu(h)
                    h = self.linear_1(h)
                    return h

            model = MLP()
            x = chainer.Variable(np.random.randn(16, 64).astype('f'))
            with SummaryWriter() as w:
                w.add_chainer_graph(model, x)

            with SummaryWriter(comment='intermediate_vars') as w:
                w.add_chainer_graph(model, x, remove_intermediate_vars=False)
