# -- Python script containing additional functions for OF_caseClass

import numpy as np

def isFloat(val):
    """function to determine if val is float"""
    try:
        float(val)
        return True
    except:
        return False
    
# -- function to find centeroid of the triangle
def centroidTriangle(vertex1, vertex2, vertex3):
    centroid = (vertex1 + vertex2 + vertex3) / 3
    return centroid

# -- function to find centeroid of the general polyhedron
def centroidGeneral(vertexes):
    centroid = np.average(vertexes,axis=0)
    return centroid