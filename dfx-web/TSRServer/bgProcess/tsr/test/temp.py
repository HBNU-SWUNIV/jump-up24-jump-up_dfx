import enum


class Kernels(enum.Enum):
    CROSS = 0
    TShape = 1
    TShape90 = 2
    TShape180 = 3
    TShape270 = 4
    LineShape = 5
    LineShape90 = 6
    LineShape180 = 7
    LineShape270 = 8
    LShape = 9
    LShape90 = 10
    LShape180 = 11
    LShape270 = 12


kernels = []


class KernelShape:
    """

    """
    def __init__(self):
        self.kernels = {
            "Cross": {
                "vector": [],
                "kernel": []
            },
            "TShape": {
                "vector": [],
                "kernel": []
            }

        }

kernel = KernelShape()
print(kernel.kernels)
