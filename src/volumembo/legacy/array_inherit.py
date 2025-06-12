import numpy as np


class c_arr(np.ndarray):
    def __new__(
        subtype,
        shape,
        ind=int,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        info=None,
    ):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)
        # set the new 'info' attribute to the value passed
        obj.ind = ind
        # Finally, we must return the newly created object:
        return obj

    def __lt__(self, other):
        index = 0
        for i in range(len(self)):
            if self[i] == 0:
                index = i
        j = (index + 1) % len(self)
        return self[j] < other[j]

    def __gt__(self, other):
        index = 0
        for i in range(len(self)):
            if self[i] == 0:
                index = i
        j = (index + 1) % len(self)
        return self[j] > other[j]

    def __eq__(self, other):
        index = 0
        for i in range(len(self)):
            if self[i] == 0:
                index = i
        j = (index + 1) % len(self)
        return self[j] == other[j]

    def __le__(self, other):
        index = 0
        for i in range(len(self)):
            if self[i] == 0:
                index = i
        j = (index + 1) % len(self)
        return self[j] <= other[j]

    def __ge__(self, other):
        index = 0
        for i in range(len(self)):
            if self[i] == 0:
                index = i
        j = (index + 1) % len(self)
        return self[j] >= other[j]

    def listed(self):
        return self.tolist()
