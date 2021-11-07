import numpy as np
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import tensorflow as tf
from scipy.stats import entropy

from AFL.TernaryPhaseMap import TernaryGridFactory

    
class GP:
    def __init__(self,pm,num_classes,dense_pts_per_row=50,use_xy=False,pm_GT=None):
    
        
        self.pm = pm
        self.xy = self.pm.comp2cart(pm.compositions)
        self.pm_GT = pm_GT
        
        self.num_classes = num_classes
        self.use_xy = use_xy
        
        self.reset_dense_grid(dense_pts_per_row)
        self.reset_GP()
        
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def reset_dense_grid(self,dense_pts_per_row):
        self.pm_dense = TernaryGridFactory(dense_pts_per_row)
        self.xy_dense = self.pm_dense.comp2cart(self.pm_dense.compositions)
    
    def reset_GP(self,kernel=None):
        
        if self.use_xy:
            data = (self.xy, self.pm.labels_ordinal) 
        else:
            data = (self.pm.compositions.values[:,[0,1]]/100.0, self.pm.labels_ordinal) 
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.1,lengthscales=0.1) 
            kernel +=  gpflow.kernels.White(variance=0.01)   
        invlink = gpflow.likelihoods.RobustMax(self.num_classes)  
        likelihood = gpflow.likelihoods.MultiClass(self.num_classes, invlink=invlink)  
        self.model = gpflow.models.VGP(
            data=data, 
            kernel=kernel, 
            likelihood=likelihood, 
            num_latent_gps=self.num_classes
        ) 
        self.loss = self.model.training_loss_closure(compile=True)
        self.trainable_variables = self.model.trainable_variables
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
        self.pm_mean = None
        self.pm_var  = None
        
        
    def reset_monitoring(self,log_dir='test/',iter_period=1):
        model_task = ModelToTensorBoard(log_dir, self.model,keywords_to_monitor=['*'])
        lml_task   = ScalarToTensorBoard(log_dir, lambda: self.loss(), "Training Loss")
        
        fast_tasks = MonitorTaskGroup([model_task,lml_task],period=iter_period)
        self.iter_monitor = Monitor(fast_tasks)
        
        image_task = ImageToTensorBoard(
            log_dir, 
            self.plot, 
            "Mean/Variance",
            fig_kw=dict(figsize=(18,6)),
            subplots_kw=dict(nrows=1,ncols=3)
        )
        slow_tasks = MonitorTaskGroup(image_task) 
        self.final_monitor = Monitor(slow_tasks)

    def plot(self,fig=None,ax=None,next_xy=None):
        if self.mean is None:
            self.predict()
        
        if ax is None:
            ax = self.mean.view.make_axes((1,3))
            
        if self.pm_GT is not None:
            self.pm_GT.plot('boundaries',ax=ax[0])
        self.pm.plot(ax=ax[0])
        
        self.mean.plot(ax=ax[1])
        self.var.plot(ax=ax[2])
        
        if next_xy is None:
            index,label,composition,next_xy = self.next()
             
        ax[0].plot(*next_xy.T,color='red',marker='x',ms=12)
        ax[1].plot(*next_xy.T,color='red',marker='x',ms=12)
        ax[2].plot(*next_xy.T,color='red',marker='x',ms=12)
        return ax
        
    def optimize(self,N,final_monitor_step=None):
        for i in tf.range(N):
            self._step(i)
        if final_monitor_step is None:
            final_monitor_step = i
        self.final_monitor(final_monitor_step)
        self.predict()
            
    @tf.function
    def _step(self,i):
        self.optimizer.minimize(self.loss,self.trainable_variables) 
        self.iter_monitor(i)
            
    def predict(self):
        if self.use_xy:
            self.y = self.model.predict_y(self.xy_dense)
        else:
            self.y = self.model.predict_y(self.pm_dense.compositions.values[:,[0,1]]/100.0)
        
        y_mean = self.y[0].numpy() 
        self.pm_mean = self.pm_dense.copy(labels=y_mean.argmax(1))
        self.pm_mean.view.cmap = 'nipy_spectral'
        
        y_var = self.y[1].numpy() 
        self.pm_var = self.pm_dense.copy(labels=y_var.sum(1))
        self.pm_var.view.cmap = 'viridis'
        
        self.pm_entropy = self.pm_dense.copy(labels=entropy(y_mean,axis=1))
        self.pm_entropy.view.cmap = 'magma'
        return self.pm_mean,self.pm_var
            
    @property
    def mean(self):
        return self.pm_mean
    
    @property
    def var(self):
        return self.pm_var
    
    @property
    def entropy(self):
        return self.pm_entropy
    
    def next(self,metric='var',nth=0,composition_check=None):
        if self.var is None:
            self.predict()
        
        metric = getattr(self,metric)
        
        #nth  =  metric.labels.shape[0]-nth-1
        while True:
            index = metric.labels.argsort()[::-1].iloc[nth]
            label = metric.labels.loc[index]
            composition = metric.compositions.loc[index]
            xy = metric.comp2cart(composition)
            if composition_check is None:
                break #all done
            elif (composition_check==composition).all(1).any():
                nth+=1
            else:
                break
            
        return index,label,composition,xy
    
    
    

    
