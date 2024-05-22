import math
class Value:
    def __init__(self,data,_children=(),_op='',label=''): 
        self.data = data
        #the driv of output node with respect to that node
        self.grad=0 #0 for no effect
        self._backpropagation=lambda: None
        #tuple of objacts
        self._prev = set(_children)
        self._op = _op 
        self.label = label

    #######operators overloading########
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data , (self,other) , '+')

        def _backpropagation():
            self.grad += out.grad
            other.grad += out.grad

        out._backpropagation =_backpropagation
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data , (self,other) , '*')

        def _backpropagation():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backpropagation =_backpropagation
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backpropagation():
            self.grad += out.data * out.grad

        out._backpropagation =_backpropagation
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only supporting int/float powers"
        out = Value(self.data**other , (self, ) , f'**{other}')

        def _backpropagation():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backpropagation =_backpropagation
        return out
    
    #######activation functions##############
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backpropagation():
            self.grad += (1 - t**2) * out.grad 

        out._backpropagation =_backpropagation
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        t = 1 / (1 + math.exp(-x))
        out = Value(t, (self,), 'sigmoid')

        def _backward():
            self.grad += t * (1-t) * out.grad

        out._backward = _backward
        return out
    
    #######backpropagation##############
    def backward(self):

        topo = []
        visited = set()
        def DFS(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    DFS(child)
                topo.append(v)
        DFS(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backpropagation()

    def __neg__(self): 
        return self * -1
    def __sub__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __radd__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        return self.__add__(other)
    
    def __rmul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        return self * other**-1

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        return other * self**-1



