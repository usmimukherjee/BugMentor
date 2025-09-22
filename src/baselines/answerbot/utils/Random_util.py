# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import random
from six.moves import range


def get_random_list(st, et, num):
    list = list(range(st, et))
    random_list = random.sample(list, num)
    return random_list
