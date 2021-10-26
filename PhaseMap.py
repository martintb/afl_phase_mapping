from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import fowlkes_mallows_score

import warnings

import pickle

class PhaseMap:
    '''Parent class controller (MVC architecture)'''
    def __init__(self,compositions,measurements,labels,metadata=None):
        self.model = PhaseMapModel(
            compositions,
            measurements,
            labels,
            metadata
        )
        self.view = PhaseMapView()
        
    def __str__(self):
        return f'<PhaseMap {self.shape[0]} pts>'
    
    def __repr__(self):
        return self.__str__()
            
    def __getitem__(self,index):
        composition = self.model.compositions.iloc[index]
        measurement = self.model.measurements.iloc[index]
        label = self.model.labels.iloc[index]
        return (composition,measurement,label)
    
    @property
    def compositions(self):
        return self.model.compositions
    
    @property
    def measurements(self):
        return self.model.measurements
    
    @property
    def labels(self):
        return self.model.labels
    
    @labels.setter
    def labels(self,labels):
        if not isinstance(labels,pd.Series):
            raise ValueError('Must pass pd.Series for labels')
        self.model.labels = labels.copy()
    
    @property
    def labels_ordinal(self):
        '''Numerical labels sorted by spatial position'''
        self.update_encoder()
        
        labels_ordinal = pd.Series(
            data=self.label_encoder.transform(
                self.labels.values.reshape(-1,1)
            ).flatten(),
            index=self.model.labels.index,
        )
        return labels_ordinal
    
    def update_encoder(self):
        labels_sorted = (
            self.compositions
            .copy()
            .set_index(self.labels)
            .sort_values(['a','b']) # sort by comp1 and then comp2
            .index
            .values
        )
        
        self.label_encoder.fit(labels_sorted.reshape(-1,1))
    
    @property
    def label_encoder(self):
        return self.model.label_encoder
    
    @property
    def shape(self):
        return self.model.compositions.shape
    
    @property
    def size(self):
        return self.model.compositions.size
    
    def copy(self,labels=None):
        if labels is None:
            labels = self.model.labels
           
        if not isinstance(labels,pd.Series):
            labels = pd.Series(labels)
            
        pm = self.__class__(
            compositions = self.model.compositions,
            measurements = self.model.measurements,
            labels = labels,
            metadata = self.model.metadata
        )
        return pm
    
    def save(self,fname):
        out_dict = {}
        out_dict['compositions'] = self.model.compositions
        out_dict['measurements'] = self.model.measurements
        out_dict['labels'] = self.model.labels
        out_dict['metadata'] = self.model.metadata
        
        if not (fname[-4:]=='.pkl'):
            fname+='.pkl'
        
        with open(fname,'wb') as f:
            pickle.dump(out_dict,f,protocol=-1)
            
    @classmethod
    def load(cls,fname):
        if not (fname[-4:]=='.pkl'):
            fname+='.pkl'
            
        with open(fname,'rb') as f:
            in_dict = pickle.load(f)
        
        pm = cls(
            compositions = in_dict['compositions'],
            measurements = in_dict['measurements'],
            labels       = in_dict['labels'],
            metadata     = in_dict['metadata'],
        )
        return pm
            
    
    def append(self,composition,measurement,label,index=None):
        if index is None:
            index = self.model.compositions.index.max()+1
            
        self.model.compositions = self.model.compositions.append(
            pd.Series( 
                data=composition,
                index=self.model.compositions.columns,
                name=index
            )
        )
            
        self.model.measurements = self.model.measurements.append(
            pd.Series( 
                data=measurement,
                index=self.model.measurements.columns,
                name=index
            )
        )
        
        self.model.labels = self.model.labels.append(
            pd.Series( 
                data=label,
                index=[index],
            )
        )
        
        # need to reset alphas if they've been set
        self.model.alphas = None
    
    def sample(self,size):
        compositions,measurements,labels = self.model.sample(size)
        pm = self.__class__(
            compositions = compositions,
            measurements = measurements,
            labels = labels,
            metadata = self.model.metadata
        )
        return pm
    
    def fms(self,other):
        labels1 = self.labels_ordinal.loc[other.labels_ordinal.index]
        labels2 = other.labels_ordinal
        return self.model.fms(labels1,labels2)
    
    
class PhaseMapModel:
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
        self.labels = labels.copy()
        
        self.label_encoder = OrdinalEncoder()
        
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = deepcopy(metadata)
            
        self.alphas = None
            
    def sample(self,size):
        compositions = self.compositions.sample(size)
        measurements = self.measurements.loc[compositions.index]
        labels = self.labels.loc[compositions.index]
        return compositions, measurements, labels
    
    def fms(self,labels1,labels2):
        return fowlkes_mallows_score(
            labels1,
            labels2,
        )
    
    def append(self,composition,label,measurement,index=None):
        if index is None:
            index = self.compositions.max()+1
            
        self.compositions = self.compositions.append(
            pd.Series( 
                data=composition,
                index=self.compositions.columns,
                name=index
            )
        )
        self.measurements = self.compositions.append(
            pd.Series( 
                data=measurement,
                index=self.measurements.columns,
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
    
    
class PhaseMapView:
    def __init__(self,cmap='jet'):
        self.cmap = cmap
        
    def make_axes(self,subplots=(1,1)):
        fig,ax = plt.subplots(*subplots,figsize=(5*subplots[1],4*subplots[0]))
        
        if (subplots[0]>1) or (subplots[1]>1):
            ax = ax.flatten()
            for cax in ax:
                cax.axis('off')
                cax.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],ls='-',color='k')
        else:
            ax.axis('off')
            ax.set(
                xlim = [0,1],
                ylim = [0,1],
            )
            ax.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],ls='-',color='k')
        return ax
    
    def scatter(self,xy,ax=None,labels=None):
        if ax is None:
            ax = self.make_axes((1,1))
        
            
        ax.scatter(*xy.T,c=labels,cmap=self.cmap,marker='.')
        return ax
    
    def lines(self,xy,ax=None,label=None):
        if ax is None:
            ax = self.make_axes((1,1))
            
        ax.plot(*xy.T,marker='None',ls=':',label=label)
        return ax