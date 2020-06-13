from __future__ import division
from __future__ import print_function

import argparse
import torch

import torch.nn as nn
from torch.autograd import Variable, gradcheck

from cuda.shift import ShiftFunction,Shift


shift_layer = Shift(channel=5,stride=2)

input = Variable(torch.ones(1,5,8,4).cuda().float(), requires_grad=True) 
out = shift_layer(input)
sum_out = torch.sum(out)
sum_out.backward()

print('*'*20 + '    input')
print(input)
print('*'*20 + '    out')
print(out)
print('*'*20 + '    input.grad')
print(input.grad)
print('*'*20 + '    shift_layer.temporal_position')
print(shift_layer.ypos)
print('*'*20 + '    shift_layer.temporal_position.grad')
print(shift_layer.ypos.grad)