from __future__ import print_function
import sys
import re
import numpy as np
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=FULL, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def fp_dec2bin(fp, format):
    """
    Converts a floating-point number to its binary representation.
    """
    if format == 16:
        num = float(fp)
        if num == 0:
            return "0000000000000000"

        sign = "0" if num > 0 else "1"

        e = 0
        num = abs(num)
        if num >= 2:
            while num >= 2:
                num /= 2
                e += 1
        elif num < 1:
            while num < 1:
                num *= 2
                e -= 1
        if e < -15:
            return "0000000000000000"
        exponent = bin(e + 15).replace("0b", "").rjust(5, "0")

        num -= 1
        if num == 0:
            return sign + exponent + "0000000000"
        mantissalist = []
        for i in range(1, 11):
            b = 2 ** -i
            if num > b:
                num -= b
                mantissalist.append("1")
            elif num == b:
                num -= b
                mantissalist.append("1")
                break
            elif num < b:
                mantissalist.append("0")

        if i == 10 and num >= 2 ** -11:
            mantissa = (bin(int('0b' + ''.join(mantissalist), 2) + 1).replace('0b', '').rjust(10, '0')[-1:-11:-1][::-1])
        else:
            mantissa = ''.join(mantissalist).ljust(10, "0")
        return sign + exponent + mantissa

    elif format == 32:
        return bin(np.float32(fp).view('I'))[2:].zfill(format)


def fp_bin2dec(fp, format):
    """
    Converts a binary floating-point representation back to its decimal form.
    """
    sign = int(fp[0])  # Convert sign bit to integer
    if format == 16:
        exp = list(fp[1:6])  # Convert exponent to a list for manipulation
        mant = list(fp[6:16])  # Convert mantissa to a list for manipulation
        exp_d = int("".join(exp), 2) - 15
        mant_d = int("".join(mant), 2) / 2 ** 10
        return (-1) ** sign * 2 ** exp_d * (1 + mant_d)
    elif format == 32:
        exp = list(fp[1:9])
        mant = list(fp[9:32])
        exp_d = int("".join(exp), 2) - 127
        mant_d = int("".join(mant), 2) / 2 ** 23
        return (-1) ** sign * 2 ** exp_d * (1 + mant_d)


def setup_seed(seed, flag):
    """
    Sets the random seed for reproducibility.
    """
    if flag == False:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


def find_samples_greedy(array, n):
    """
    Select PBEDC samples using greedy search strategy.
    """
    total_errors = array.shape[1]
    index_list = []
    coverage_list = []
    uncovered_mask = np.ones(total_errors, dtype=bool)  # Mask for uncovered errors

    for _ in range(n):
        # Compute uncovered errors detected by each sample
        coverage = np.sum(array[:, uncovered_mask], axis=1)

        # Select the sample with maximum uncovered coverage
        max_index = np.argmax(coverage)
        index_list.append(max_index)

        # Update uncovered errors
        new_covered = np.where(array[max_index, :] == 1)[0]
        uncovered_mask[new_covered] = False
        coverage_list.append(np.sum(~uncovered_mask))

        # Stop if all errors are covered
        if np.all(~uncovered_mask):
            break

    uncovered_errors = np.sum(uncovered_mask)
    return index_list, coverage_list, uncovered_errors

def find_samples_max_coverage(array, n):
    """
    Select PBEDC samples using max overlapped coverage strategy.
    """
    total_errors = array.shape[1]
    index_list = []
    coverage_list = []
    uncovered_mask = np.ones(total_errors, dtype=bool)  # Mask for uncovered errors

    # Precompute coverage for all samples
    sample_coverage = np.sum(array, axis=1)

    # Sort samples by their overall coverage in descending order
    sorted_indices = np.argsort(-sample_coverage)

    for i in range(n):
        # Select the next sample in sorted order
        max_index = sorted_indices[i]
        index_list.append(max_index)

        # Update uncovered errors
        new_covered = np.where(array[max_index, :] == 1)[0]
        uncovered_mask[new_covered] = False
        coverage_list.append(np.sum(~uncovered_mask))

        # Stop if all errors are covered
        if np.all(~uncovered_mask):
            break

    uncovered_errors = np.sum(uncovered_mask)
    return index_list, coverage_list, uncovered_errors