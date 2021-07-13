from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import fowlkes_mallows_score

from AFL.PhasePlotter import TernaryPhasePlotter
from AFL.Ternary import Ternary
import warnings


#### Need MVC!! 

class PhaseMap:
    def __init__(self,compositions,measurements,labels,metadata=None):
        if not isinstance(compositions,pd.DataFrame):
            raise ValueError('Must pass pd.Dataframe for composition')
            
        if not isinstance(measurements,pd.DataFrame):
            raise ValueError('Must pass pd.Dataframe for measurements')
            
        if not isinstance(labels,pd.Series):
            raise ValueError('Must pass pd.Series for labels')
            
        self.compositions = compositions.copy()
        self.measurements = measurements.copy()
        
        #copy labels and then premptively ordinally encode 
        # even if already encoded, but store original
        self.labels_orig = labels.copy()
        self.labels      = labels.copy()
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
    
    def copy(self,labels=None):
        if labels is None:
            labels = self.labels
           
        if not isinstance(labels,pd.Series):
            labels = pd.Series(labels)
            
        pm = self.__class__(
            compositions = self.compositions,
            measurements = self.measurements,
            labels = labels,
            metadata = self.metadata
        )
        return pm
    
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
        
        if hasattr(self,alphas):
            self.alphas = None
  
    
        
class TernaryPhaseMap(PhaseMap,Ternary):
    def __init__(self,compositions,measurements,labels,metadata=None):
        super().__init__(compositions,measurements,labels,metadata)
        self.alphas = None
        self.plot = TernaryPhasePlotter(self.compositions,self.labels,self.alphas)
        

    def __str__(self):
        return f'<TernaryPhaseMap {self.shape[0]} pts>'
            
    def trace_boundaries(self,alpha=10,drop_phases=None):
        import alphashape
        if drop_phases is None:
            drop_phases = []
        
        self.alphas = {}
        for phase in self.labels.unique():
            if phase in drop_phases:
                continue
            mask = (self.labels==phase)
            xy = self.comp2cart(self.compositions.loc[mask])
            self.alphas[phase] = alphashape.alphashape(xy,alpha) 
        self.plot.alphas = self.alphas
         
            
    def locate(self,composition):
        from shapely.geometry import Point
        composition = np.array(composition)
        
        if self.alphas is None:
            raise ValueError('Must call trace_boundaries before locate')
            
        point = Point(*self.comp2cart(composition))
        locations = {}
        for phase,alpha in self.alphas.items():
            if alpha.contains(point):
                locations[phase] = True
            else:
                locations[phase] = False
                
        if sum(locations.values())>1:
            warnings.warn('Location in multiple phases. Phases likely overlapping')
        phases = [key for key,value in locations.items() if value]
            
        return phases
    

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
    labels       =  pd.Series(np.zeros_like(compositions[0]))
    measurements =  pd.DataFrame(np.zeros_like(compositions[0]))
    pm = TernaryPhaseMap(compositions,measurements,labels,metadata)
    return pm
    