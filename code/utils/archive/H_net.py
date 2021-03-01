import numpy as np

def get_H_net_parameters(img=None):
    a, b, c, d, e, f = 1, 0, 0, 1, 0, 0
    return a, b, c, d, e, f

def get_H_net_transformation(img=None):
    a, b, c, d, e, f = get_H_net_parameters(img)
    transformation = np.array([[a, b, c], [0, d, e], [0, f, 1]])
    return transformation
