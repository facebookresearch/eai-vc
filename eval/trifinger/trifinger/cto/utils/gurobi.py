import gurobipy as gp
import numpy as np


def sum(vars):
    result = 0
    for var in vars:
        result += var
    return result


def abs_diff_binary(x, y):

    result = MVar(x.model, shape=x.shape, vtype=x.vtype)
    model = x.model
    for m in range(x.nrows):
        for n in range(x.ncols):
            model.addConstr(result[m, n] <= x[m, n] + y[m, n])
            model.addConstr(result[m, n] <= x[m, n] - y[m, n])
            model.addConstr(result[m, n] <= y[m, n] - x[m, n])
            model.addConstr(result[m, n] <= 2 - x[m, n] - y[m, n])
    return result


class MVar(object):
    def __init__(self, model, shape=None, var=None, vtype=None):
        self.model = model
        self.ndim = 2  # always a 2D matrix
        self.vtype = vtype

        if shape is not None:
            self.nrows, self.ncols = shape
            var = []
            for m in range(self.nrows):
                row = []
                for n in range(self.ncols):
                    if vtype is not None:
                        row.append(model.addVar(lb=float("-inf"), vtype=vtype))
                    else:
                        row.append(model.addVar(lb=float("-inf")))
                var.append(tuple(row))
            self.var = tuple(var)
        if var is not None:
            if isinstance(var, list) or isinstance(var, tuple):
                self.nrows = len(var)
                for m in range(len(var)):
                    if isinstance(var[m], list) or isinstance(var[m], tuple):
                        self.ncols = len(var[0])  # assuming same length
                        self.var = tuple(var)
                    else:
                        self.ncols = 1
                        self.var = tuple((tuple([var[i]]) for i in range(self.nrows)))
                        break
            else:
                # assuming scalar
                self.var = tuple((tuple([var])))
                self.nrows = self.ncols = 1

        self.size = self.nrows * self.ncols

    def __repr__(self):
        return "Gurobi Matrix of {nrows} rows and {ncols} cols".format(
            nrows=self.nrows, ncols=self.ncols
        )

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return MVar(self.model, var=self.var[index])
        elif isinstance(index, tuple):
            row, col = index
            if isinstance(row, int) and isinstance(col, int):
                return self.var[row][
                    col
                ]  # do not return a MVar object for a single element
            elif isinstance(row, int) and isinstance(col, slice):
                return MVar(self.model, var=self.var[row][col])
            elif isinstance(row, slice) and isinstance(col, (int, slice)):
                return MVar(self.model, var=[r[col] for r in self.var[row]])
            else:
                raise NotImplementedError("Invalid argument type")
        else:
            raise NotImplementedError("Invalid argument type")

    def __len__(self):
        return self.nrows

    @property
    def shape(self):
        return (self.nrows, self.ncols)

    def sum(self, axis=None):
        if axis == 0:
            return self._sum_rows()
        elif axis == 1:
            return self._sum_cols()
        elif axis is None:
            return self._sum_cols().sum(axis=0)[0, 0]

    def _sum_rows(self):
        # does not retain the shape of a row vector
        # vectors are always assumeed to be column vectors
        summation = []
        for n in range(self.ncols):
            summation_n = 0
            for m in range(self.nrows):
                summation_n += self[m, n]
            summation.append(summation_n)
        return MVar(self.model, var=summation)

    def _sum_cols(self):
        summation = []
        for m in range(self.nrows):
            summation_m = 0
            for n in range(self.ncols):
                summation_m += self[m, n]
            summation.append(summation_m)
        return MVar(self.model, var=summation)

    def __neg__(self):
        result = []
        for i in range(self.nrows):
            row = [0] * self.ncols
            result.append(row)
        for m in range(self.nrows):
            for n in range(self.ncols):
                result[m][n] = -self[m, n]

        return MVar(self.model, var=result)

    def __eq__(self, rhs):
        constr = []
        if hasattr(rhs, "__len__"):

            if rhs.shape == self.shape:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] == rhs[m, n])
            elif (
                (len(rhs.shape) == 1 or rhs.shape[0] == 1)
                and self.ncols == 1
                and self.nrows == sum(rhs.shape)
            ):
                rhs = rhs.reshape((-1, 1))
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] == rhs[m, n])
            elif rhs.shape[0] == 1 and rhs.shape[1] == 1:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] == rhs[0, 0])
            else:
                raise ValueError("Dimension mistach")
        else:
            for m in range(self.nrows):
                for n in range(self.ncols):
                    constr.append(self[m, n] == rhs)
        return constr

    def __le__(self, rhs):
        constr = []
        if hasattr(rhs, "__len__"):

            if rhs.shape == self.shape:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] <= rhs[m, n])
            elif (
                (len(rhs.shape) == 1 or rhs.shape[0] == 1)
                and self.ncols == 1
                and self.nrows == sum(rhs.shape)
            ):
                rhs = rhs.reshape((-1, 1))
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] <= rhs[m, n])
            elif rhs.shape[0] == 1 and rhs.shape[1] == 1:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] <= rhs[0, 0])
            else:
                raise ValueError("Dimension mistach")
        else:
            for m in range(self.nrows):
                for n in range(self.ncols):
                    constr.append(self[m, n] <= rhs)
        return constr

    def __ge__(self, rhs):
        return (-self).__le__(-rhs)

    def __lt__(self, rhs):
        constr = []
        if hasattr(rhs, "__len__"):

            if rhs.shape == self.shape:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] < rhs[m, n])
            elif (
                (len(rhs.shape) == 1 or rhs.shape[0] == 1)
                and self.ncols == 1
                and self.nrows == sum(rhs.shape)
            ):
                rhs = rhs.reshape((-1, 1))
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] < rhs[m, n])
            elif rhs.shape[0] == 1 and rhs.shape[1] == 1:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        constr.append(self[m, n] < rhs[0, 0])
            else:
                raise ValueError("Dimension mistach")
        else:
            for m in range(self.nrows):
                for n in range(self.ncols):
                    constr.append(self[m, n] < rhs)
        return constr

    def __gt__(self, rhs):
        return (-self).__lt__(-rhs)

    def __matmul__(self, b):
        a = self

        nrows_a = len(a)
        ncols_a = len(a[0])
        nrows_b = len(b)
        ncols_b = len(b[0])

        if ncols_a != nrows_b:
            raise ValueError("Dimension mistach")

        result = []
        for i in range(nrows_a):
            row = [0] * ncols_b
            result.append(row)

        for i in range(nrows_a):
            for j in range(ncols_b):
                for k in range(ncols_a):
                    result[i][j] += a[i, k] * b[k, j]

        return MVar(self.model, var=result)

    def __rmatmul__(self, a):
        b = self

        nrows_a = len(a)
        ncols_a = len(a[0])
        nrows_b = len(b)
        ncols_b = len(b[0])

        if ncols_a != nrows_b:
            raise ValueError("Dimension mistach")

        result = []
        for i in range(nrows_a):
            row = [0] * ncols_b
            result.append(row)

        for i in range(nrows_a):
            for j in range(ncols_b):
                for k in range(ncols_a):
                    result[i][j] += a[i, k] * b[k, j]

        return MVar(self.model, var=result)

    def __add__(self, b):
        a = self
        nrows_a = len(a)
        ncols_a = len(a[0])

        result = []
        for i in range(nrows_a):
            row = [0] * ncols_a
            result.append(row)

        if hasattr(b, "__len__"):
            if b.shape == a.shape:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        result[m][n] = a[m, n] + b[m, n]
            elif len(b.shape) == 1 and self.ncols == 1:
                for m in range(self.nrows):
                    result[m][0] = a[m, 0] + b[m]
            else:
                raise ValueError("Dimension mistach")
        else:
            for m in range(self.nrows):
                for n in range(self.ncols):
                    result[m][n] = a[m, n] + b

        return MVar(self.model, var=result)

    def __radd__(self, a):
        return self.__add__(a)

    def __sub__(self, b):
        return self + (-b)

    def __rsub__(self, a):
        return (-self).__add__(a)

    def __mul__(self, b):
        a = self
        nrows_a = len(a)
        ncols_a = len(a[0])

        if nrows_a == 1 and ncols_a == 1:
            if hasattr(b, "__len__"):
                nrows_b = len(b)
                if hasattr(b[0], "__len__"):
                    ncols_b = len(b[0])
                    result = []
                    for i in range(nrows_b):
                        row = [0] * ncols_b
                        result.append(row)
                    for m in range(nrows_b):
                        for n in range(ncols_b):
                            result[m][n] = a[0, 0] * b[m, n]
                else:
                    result = [0] * nrows_b
                    for m in range(nrows_b):
                        result[m] = a[0, 0] * b[m]
            else:
                result = [[a[0, 0] * b]]
        else:
            result = []
            for i in range(nrows_a):
                row = [0] * ncols_a
                result.append(row)

            if hasattr(b, "__len__"):
                if b.shape == a.shape:
                    for m in range(self.nrows):
                        for n in range(self.ncols):
                            result[m][n] = a[m, n] * b[m, n]
                else:
                    raise ValueError("Dimension mistach")
            else:
                for m in range(self.nrows):
                    for n in range(self.ncols):
                        result[m][n] = a[m, n] * b

        return MVar(self.model, var=result)

    def __rmul__(self, a):
        return self * a

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        a, b = inputs
        if ufunc.__name__ == "matmul" and method == "__call__":
            return b.__rmatmul__(a)
        elif ufunc.__name__ == "multiply" and method == "__call__":
            return b.__rmul__(a)
        elif ufunc.__name__ == "add" and method == "__call__":
            return b.__radd__(a)
        elif ufunc.__name__ == "subtract" and method == "__call__":
            return b.__rsub__(a)

    @property
    def T(self):
        result = []
        for i in range(self.ncols):
            row = [0] * self.nrows
            result.append(row)

        for m in range(self.nrows):
            for n in range(self.ncols):
                result[n][m] = self[m, n]

        return MVar(self.model, var=result)

    @property
    def value(self):
        result = np.zeros((self.nrows, self.ncols))

        for m in range(self.nrows):
            for n in range(self.ncols):
                if isinstance(self[m, n], gp.LinExpr):
                    result[m, n] = self[m, n].getValue()
                else:
                    result[m, n] = self[m, n].x
        return np.squeeze(result)

    def square(self):
        if self.nrows == 1:
            return (self @ self.T)[0, 0]
        elif self.ncols == 1:
            return (self.T @ self)[0, 0]
        else:
            result = []
            for i in range(self.nrows):
                row = [0] * self.ncols
                result.append(row)
            for m in range(self.nrows):
                for n in range(self.ncols):
                    result[m][n] = self[m, n] ** 2

            return MVar(self.model, var=result)

    def abs(self):
        result = MVar(self.model, shape=self.shape)
        for m in range(self.nrows):
            for n in range(self.ncols):
                self.model.addGenConstrAbs(result[m, n], self[m, n])
        return result
