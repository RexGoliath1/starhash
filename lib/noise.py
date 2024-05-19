import numpy as np
import random

def add_sp_noise(image, prob = 0.05):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j][k] = 0
                elif rdn > thres:
                    output[i][j][k] = 255
                else:
                    output[i][j][k] = image[i][j][k]
    return output
