from pprint import pprint

import numpy as np

kernel_7x7_cross = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=np.uint8)

kernel_7x7_T = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
], dtype=np.uint8)

kernel_7x7_lines = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

kernel_7x7_L = np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

"""
direction
0 0 0 0
U D L R
"""
kernels = [
    {
        "shape": "cross",
        "kernel": kernel_7x7_cross,
        "direction": 0b1111
    },
    {
        "shape": "TShape0",
        "kernel": kernel_7x7_T,
        "direction": 0b0111
    },
    {
        "shape": "TShape90",
        "kernel": np.rot90(kernel_7x7_T, k=1),
        "direction": 0b1101
    },
    {
        "shape": "TShape180",
        "kernel": np.rot90(kernel_7x7_T, k=2),
        "direction": 0b1011
    },
    {
        "shape": "TShape270",
        "kernel": np.rot90(kernel_7x7_T, k=3),
        "direction": 0b1110
    },
    {
        "shape": "LineShape0",
        "kernel": kernel_7x7_lines,
        "direction": 0b1000
    },
    {
        "shape": "LineShape90",
        "kernel": np.rot90(kernel_7x7_lines, k=1),
        "direction": 0b0010
    },
    {
        "shape": "LineShape180",
        "kernel": np.rot90(kernel_7x7_lines, k=2),
        "direction": 0b0100
    },
    {
        "shape": "LineShape270",
        "kernel": np.rot90(kernel_7x7_lines, k=3),
        "direction": 0b0001
    },
    {
        "shape": "LShape0",
        "kernel": kernel_7x7_L,
        "direction": 0b1001
    },
    {
        "shape": "LShape90",
        "kernel": np.rot90(kernel_7x7_L, k=1),
        "direction": 0b1010
    },
    {
        "shape": "LShape180",
        "kernel": np.rot90(kernel_7x7_L, k=2),
        "direction": 0b0110
    },
    {
        "shape": "LShape270",
        "kernel": np.rot90(kernel_7x7_L, k=3),
        "direction": 0b0101
    }
]

if __name__ == '__main__':
    for idx, kernel in enumerate(kernels):
        print(kernel["shape"])
        print(kernel["kernel"])
        print(format(kernel["direction"], '04b'))
        print("========================")

