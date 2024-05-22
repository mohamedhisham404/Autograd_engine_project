from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, filename="graph"):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})  

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # For any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label="{%s | data %.4f | grad %.4f}" % (n.label ,n.data , n.grad), shape='record')
        if n._op:
            # If this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # And connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # Connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    output_path = dot.render(filename)
    print(f"Graph saved to {output_path}")

    img = mpimg.imread('graph.png')
    imgplot = plt.imshow(img)
    plt.show()
    

    return dot

