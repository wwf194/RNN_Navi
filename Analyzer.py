



import numpy as np
from sklearn.decomposition import PCA
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#from sklearn.decomposition import PCA

class Analyzer:
    def __init__(self):
        return

class Analyzer_PCA:
    def __init__(self, dim_num=3):
        self.pca = PCA(n_components=dim_num)
    def fit(self, data): #data:[sample_num, feature_size]
        self.pca.fit(data)
    def visualize_traj(self, trajs): #data:[traj_num][traj_length, N_num]
        fig = plt.figure()
        plt.title("Neural Trajectories")
        ax = fig.gca(projection='3d')
        #colors = [c for c in "bgrcmyk"]
        #color_index = 0
        mpl.rcParams['legend.fontsize'] = 10
        for traj in trajs:
            traj_trans = self.pca.transform(traj) #[dim_num, traj_length]
            ax.plot(traj_trans[:,0], traj_trans[:, 1], traj_trans[:, 2], label='parametric curve')
            #ax.legend()
        plt.show()
        plt.savefig("./trajs_PCA3d.png")

