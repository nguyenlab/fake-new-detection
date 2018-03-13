import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import numpy as np

def generate_visualize_array(p):
    aa = np.array(p)
    aa = np.reshape(aa, newshape= [1, -1])
    return aa

def visualize(p, text):
    """
    p = [0.1, 0.3, 0, 0.6, 0]
    text = ['a', 'b', 'c', 'd', 'e']
    :param p: 
    :param text: 
    :return: 
    """
    p = 1 - p
    y_bar = ['']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # cax = ax.matshow(samplemat(5),cmap=cm.gray)
    cax = ax.matshow(generate_visualize_array(p), cmap=cm.gray)
    fig.colorbar(cax, orientation="horizontal")

    ax.set_xticklabels([''] + text)
    ax.set_yticklabels([''] + y_bar)

    plt.show()

attend_weights = np.loadtxt("atention_weights.txt", dtype= np.float32)

for idx, w in enumerate(attend_weights):
    if idx == 709: # dong 31
        non = np.nonzero(w)
        len_non = len(non[0])
        a = w[: len_non]

        #text = ['a']* len_non
        #text = ['eighty', 'percent', 'wall', 'street', 'executives', 'spouses', 'donations', 'go', 'democrats']
        #text = ['says', 'John', 'Mccain', 'done', 'nothing', 'help', 'vets']
        #text = ['united', 'states', 'low', 'voter', 'turnout', 'rate']
        #text = ['almost', 'every', 'state', 'offered', 'insurance', 'plan', 'health', 'exchange', 'cover', 'abortion']
        text = ['says', 'would', 'first', 'cpa', 'serve', 'texas', 'comptroller']
        #text = ['says', 'Thom', 'Tillis', 'gives', 'tax', 'breaks', 'yacht', 'jet', 'owners']
        print(a)
        print(text)
        visualize(a, text)
        exit()
