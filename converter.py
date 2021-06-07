"""
Convert the binary representation to a SDF-like representation
(for the purpose of unifying shape representation in optimization)

Author(s): Wei Chen (wchen459@gmail.com)
"""

def convert_points(points_int, rez):
    return (points_int+0.5)/rez-0.5

def convert_values(values):
    return values - .5

def convert_gradients(gradients):
    return -gradients