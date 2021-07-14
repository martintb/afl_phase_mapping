from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import fowlkes_mallows_score

import warnings
from AFL.PhaseMap import PhaseMap,PhaseMapModel,PhaseMapView


class TernaryPhaseMap(PhaseMap):
    def __init__(self,compositions,measurements,labels,metadata=None):
        self.model = PhaseMapModel(
            compositions,
            measurements,
            labels,
            metadata
        )
        self.view = PhaseMapView()
        
    def __str__(self):
        return f'<TernaryPhaseMap {self.shape[0]} pts>'
    
    def comp2cart(self,compositions=None):
        '''Ternary composition to Cartesian cooridate'''
        if compositions is None:
            compositions = self.model.compositions
            
        try:
            #Assume pandas
            t = compositions.values.copy()
        except AttributeError:
            # Assume numpy
            t = compositions.copy()
        
        if t.ndim==1:
            t = np.array([t])
            
        # Convert ternary data to cartesian coordinates.
        xy = np.zeros((t.shape[0],2))
        xy[:,1] = t[:,1]*np.sin(60.*np.pi / 180.) / 100.
        xy[:,0] = t[:,0]/100. + xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
        return xy
    
    def trace_boundaries(self,alpha=10,drop_phases=None):
        import alphashape
        if drop_phases is None:
            drop_phases = []
        
        self.model.alphas = {}
        for phase in self.model.labels.unique():
            if phase in drop_phases:
                continue
            mask = (self.model.labels==phase)
            xy = self.comp2cart(self.model.compositions.loc[mask])
            self.model.alphas[phase] = alphashape.alphashape(xy,alpha) 
         
            
    def locate(self,composition,plot=False,ax=None):
        from shapely.geometry import Point
        composition = np.array(composition)
        
        if self.model.alphas is None:
            raise ValueError('Must call trace_boundaries before locate')
            
        point = Point(*self.comp2cart(composition))
        locations = {}
        for phase,alpha in self.model.alphas.items():
            if alpha.contains(point):
                locations[phase] = True
            else:
                locations[phase] = False
                
        if sum(locations.values())>1:
            warnings.warn('Location in multiple phases. Phases likely overlapping')
            
        phases = [key for key,value in locations.items() if value]
        
        if plot:
            ax = self.plot('boundaries',ax=ax)
            plt.legend()
            plt.gca().plot(point.x,point.y,color='red',marker='x',markersize=8)
            
            return phases,ax
        else:
            return phases
    
    def plot(self,kind=None,ax=None):
        if kind is None:
            xy = self.comp2cart(self.model.compositions)
            ax = self.view.scatter(
                xy=xy,
                ax=ax,
                labels=self.labels_ordinal
            )
        elif kind is 'boundaries':
            if self.model.alphas is None:
                self.trace_boundaries()
            for name,alpha in self.model.alphas.items():
                xy = np.vstack(alpha.boundary.xy).T
                ax = self.view.lines(xy,ax,label=name)
        else:
            raise ValueError('Unrecognized plot type')
        return ax
    
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

    

