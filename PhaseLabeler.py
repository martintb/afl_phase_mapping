import numpy as np
import copy
import scipy.spatial

from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance as dist
from collections import Counter



class PhaseLabeler:
    def __init__(self,params=None):
        self.labels = None
        if params is None:
            self.params = {}
        else:
            self.params = params
            
    def copy(self):
        return copy.deepcopy(self)
        
    def __getitem__(self,index):
        return self.labels[index]
    
    def __array__(self,dtype=None):
        return np.array(self.labels).astype(dtype)

    def remap_labels_by_count(self):
        label_map ={}
        for new_label,(old_label,_) in enumerate(sorted(Counter(self.labels).items(),key=lambda x: x[1],reverse=True)):
            label_map[old_label]=new_label
        self.labels = list(map(label_map.get,self.labels))
        
    def label(self):
        raise NotImplementedError('Sub-classes must implement label!')
        
class PhaseLabelerGMM(PhaseLabeler):
    def label(self,X,**params):
        if params:
            self.params.update(params)
        self.clf = GaussianMixture(self.params['n_cluster'])
        self.clf.fit(X)
        self.labels = self.clf.predict(X)
        
class PhaseLabelerSC(PhaseLabeler):
    def label(self,X,**params):
        if params:
            self.params.update(params)
            
        self.clf(**self.params)
        # n_clusters=N,  
        # affinity = 'precomputed',  
        # assign_labels="discretize",  
        # random_state=0,  
        # n_init = 1000
        self.clf.fit(X)
        self.labels = self.clf.labels_
        