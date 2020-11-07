class A:
    def __init__(self):
        pass

    def add(self, x, y):
        return x + y


class B(A):
    def __init__(self):
        pass

    def add(self, x, y):
        res = super(B, self).add(x, y)
        return res


a = A()
b = B()
print(b.add(1, 2))
