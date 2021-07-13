from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AFL.Ternary import Ternary

class PhasePlotter:
    def __init__(self,compositions,labels,cmap='jet',alphas=None):
        #should be references
        self.compositions = compositions
        self.labels = labels
        self.cmap = cmap
        self.alphas = alphas
    
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
    
    def scatter(self,xy,fig=None,ax=None):
        if ax is None:
            ax = self.make_axes((1,1))
            
        ax.scatter(*xy.T,c=self.labels,cmap=self.cmap,marker='.')
        return ax
    
    def lines(self,xy,fig=None,ax=None,label=None):
        if ax is None:
            ax = self.make_axes((1,1))
            
        ax.plot(*xy.T,marker='None',ls=':',label=label)
        return ax
    
        
class TernaryPhasePlotter(PhasePlotter,Ternary):
        
    def scatter(self,xy=None,fig=None,ax=None):
        if xy is None:
            xy = self.comp2cart(self.compositions)
        ax = super().scatter(xy,fig,ax)
        return ax
    
    def boundaries(self,alphas=None,fig=None,ax=None):
        if (alphas is None) and (self.alphas is None):
            raise ValueError('Need to pass alphas or call trace_boundaries')
        elif (alphas is None):
            alphas = self.alphas
            
        for name,alpha in alphas.items():
            xy = np.vstack(alpha.boundary.xy).T
            ax = super().lines(xy,fig,ax,label=name)
        return ax
        
            
    
