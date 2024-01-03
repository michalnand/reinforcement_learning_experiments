import numpy
import matplotlib.pyplot as plt



if __name__ == "__main__":
 
    with open("models/ppo_csnd_10_0/trained/explored_map.npy", "rb") as f:
        explored_map = numpy.load(f) 

    im = numpy.log(1.0 + explored_map)
    #im = im/(im.max() + 10**-6)

    plt.imshow(im)
    plt.show()