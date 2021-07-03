from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TernaryPhasePlotter:
    def __init__(self,compositions,labels,cmap='jet'):
        #should be references
        self.compositions = compositions
        self.labels = labels
        self.cmap = cmap
    
    def _comp2cart(self):
        t = self.compositions.values.copy()
        # Convert ternary data to cartesian coordinates.
        xy = np.zeros((t.shape[0],2))
        xy[:,1] = t[:,1]*np.sin(60.*np.pi / 180.) / 100.
        xy[:,0] = t[:,0]/100. + xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
        return xy
    
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
        
    def plot(self,fig=None,ax=None,field=False):
        if ax is None:
            ax = self.make_axes((1,1))
        xy = self._comp2cart()
        if field:
            ax.scatter(*xy.T,c=self.labels,cmap=self.cmap)
        else:
            ax.scatter(*xy.T,c=self.labels,cmap=self.cmap,marker='.')
        return ax