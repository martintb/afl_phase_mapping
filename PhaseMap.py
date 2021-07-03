from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import fowlkes_mallows_score

from AFL.PhasePlotter import TernaryPhasePlotter

class PhaseMap:
    def __init__(self,compositions,measurements,labels,metadata=None):
        self.compositions = deepcopy(compositions)
        self.measurements = deepcopy(measurements)
        
        #copy labels and then premptively ordinally encode 
        # even if already encoded, but store original
        self.labels_orig = deepcopy(pd.Series(labels))
        self.labels      = deepcopy(pd.Series(labels))
        self.labels.iloc[:] = (
            OrdinalEncoder().fit_transform(self.labels.values.reshape(-1,1))
        ).flatten()
        
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = deepcopy(metadata)
            
    def __str__(self):
        return f'<PhaseMap {self.shape[0]} pts>'
    
    def __repr__(self):
        return self.__str__()
            
    def __getitem__(self,index):
        composition = self.compositions.iloc[index]
        measurement = self.measurements.iloc[index]
        label = self.labels.iloc[index]
        return (composition,measurement,label)
    
    def fms(self,other):
        return fowlkes_mallows_score(
            self.labels.loc[other.labels.index],
            other.labels
        )
    
    @property
    def shape(self):
        return self.compositions.shape
    
    @property
    def size(self):
        return self.compositions.size
    
    def sample(self,size=None,ids=None):
        if (size is None) and (ids is None):
            raise ValueError('Must specify random sample size or specific ids')
        elif (ids is None):
            compositions = self.compositions.sample(size)
            measurements = self.measurements.loc[compositions.index]
            labels = self.labels.loc[compositions.index]
        else:
            compositions,measurements,labels = self[ids]
            
        pm = self.__class__(
            compositions = compositions,
            measurements = measurements,
            labels = labels,
            metadata = self.metadata
        )
        return pm
            
    
    def append(self,composition,label,index=None):
        if index is None:
            index = self.compositions.max()+1
        self.compositions = self.compositions.append(
            pd.Series( 
                data=composition,
                index=['a','b','c'],
                name=index
            )
        )
        self.labels = self.labels.append(
            pd.Series( 
                data=label,
                index=['value'],
                name=index
            )
        )
    
    def copy(self,labels=None):
        if labels is None:
            labels = self.labels
            
        pm = self.__class__(
            compositions = self.compositions,
            measurements = self.measurements,
            labels = labels,
            metadata = self.metadata
        )
        return pm
        
        
class TernaryPhaseMap(PhaseMap):
    def __init__(self,compositions,measurements,labels,metadata=None):
        super().__init__(compositions,measurements,labels,metadata)
        self.plotter = TernaryPhasePlotter(self.compositions,self.labels)
        
    def __str__(self):
        return f'<TernaryPhaseMap {self.shape[0]} pts>'

def TernaryGridFactory(pts_per_row=50,basis=100,metadata=None):
    compositions = []
    eps = 1e-9 #floating point comparison bound
    for i in np.linspace(0,1.0,pts_per_row):
        for j in np.linspace(0,1.0,pts_per_row):
            if i+j>(1+eps):
                continue

            k = 1.0 - i - j
            if k<(0-eps):
                continue

            compositions.append([i*basis,j*basis,k*basis])
    compositions = pd.DataFrame(compositions)
    labels =  pd.Series(np.zeros_like(compositions[0]))
    measurements =  pd.DataFrame(np.zeros_like(compositions[0]))
    pm = TernaryPhaseMap(compositions,measurements,labels,metadata)
    return pm
        
    