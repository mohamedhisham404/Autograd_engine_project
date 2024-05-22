from NN import MLP
from visualize import draw_dot
from auto_grad_engine import Value

def test1():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f

    print(f'{g.data:.4f}')#24.7041
    g.backward() 
    draw_dot(g)
    print(f'{a.grad:.4f}')#145.7755
    print(f'{b.grad:.4f}')#638.6356

def test2():#not finnished
    x = [2.0, 3.0]
    NN = MLP(2, [4, 4, 1])

    xs=[[2.0, 3.0, -1.0], 
    [4.0, 5.0, 0.5], 
    [6.0, 7.0, 1.0],
    [1.0, 1.0, -1.0]]

    ys=[[1.0], 
    [-1.0], 
    [-1.0],
    [1.0]]

    ypred = [NN(x) for x in xs]
    print(ypred)

if __name__ == '__main__':
    test1()

