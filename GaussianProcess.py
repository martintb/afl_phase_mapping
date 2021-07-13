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

from AFL.TernaryPhaseMap import TernaryGridFactory

    
class GP:
    def __init__(self,pm,num_classes,dense_pts_per_row=50,use_xy=False):
    
        
        self.pm = pm
        self.xy = self.pm.comp2cart(pm.compositions)
        
        self.num_classes = num_classes
        self.use_xy = use_xy
        
        self.reset_dense_grid(dense_pts_per_row)
        self.reset_GP()
        
        self.pm_mean = None
        self.pm_var  = None
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def reset_dense_grid(self,dense_pts_per_row):
        self.pm_dense = TernaryGridFactory(dense_pts_per_row)
        self.xy_dense = self.pm_dense.comp2cart(self.pm_dense.compositions)
    
    def reset_GP(self,kernel=None):
        
        if self.use_xy:
            data = (self.xy, self.pm.labels) 
        else:
            data = (self.pm.compositions.values[:,[0,1]]/100.0, self.pm.labels) 
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.5,lengthscales=0.5) 
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
        
        
    def reset_monitoring(self,log_dir='test/',iter_period=1):
        model_task = ModelToTensorBoard(log_dir, self.model,keywords_to_monitor=['*'])
        lml_task   = ScalarToTensorBoard(log_dir, lambda: self.loss(), "Training Loss")
        
        fast_tasks = MonitorTaskGroup([model_task,lml_task],period=iter_period)
        self.iter_monitor = Monitor(fast_tasks)
        
        image_task = ImageToTensorBoard(
            log_dir, 
            self.plot, 
            "Mean/Variance",
            fig_kw=dict(figsize=(12,6)),
            subplots_kw=dict(nrows=1,ncols=2)
        )
        slow_tasks = MonitorTaskGroup(image_task) 
        self.final_monitor = Monitor(slow_tasks)

    def plot(self,fig,ax=None):
        self.predict()
        
        if ax is None:
            ax = self.mean.plot.make_axes((1,2))
        else:
            for cax in ax:
                cax.axis('off')
                cax.set(
                    xlim = [0,1],
                    ylim = [0,1],
                )
                cax.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],ls='-',color='k')
        self.mean.plot(ax=ax[0])
        self.var.plot(ax=ax[1])
        
    def optimize(self,N):
        for i in tf.range(N):
            self._step(i)
        self.final_monitor(i)
            
    # @tf.function
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
        self.pm_mean.view.cmap = 'jet'
        
        y_var = self.y[1].numpy() 
        self.pm_var = self.pm_dense.copy(labels=y_var.sum(1))
        self.pm_var.view.cmap = 'viridis'
        return self.pm_mean,self.pm_var
            
    @property
    def mean(self):
        return self.pm_mean
    
    @property
    def var(self):
        return self.pm_var
    

    
