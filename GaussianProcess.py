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

from AFL.PhaseMap import TernaryGridFactory

# @tf.function
def _step(i,optimizer,loss,trainable_variables):
    optimizer.minimize(loss,trainable_variables) # run the optimization
    #monitor(i)
    
class GP:
    def __init__(self,pm,num_classes,dense_pts_per_row=50,use_xy=False):
    
        
        self.pm = pm
        self.xy = self.pm.plotter._comp2cart()
        
        self.num_classes = num_classes
        self.use_xy = use_xy
        
        self.reset_dense_grid(dense_pts_per_row=50)
        self.reset_GP()
        
        self.pm_mean = None
        self.pm_var  = None
        
    def reset_dense_grid(self,dense_pts_per_row=50):
        self.pm_dense = TernaryGridFactory(dense_pts_per_row)
        self.xy_dense = self.pm_dense.plotter._comp2cart()
    
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
        
    def optimize(self,N):
        for i in tf.range(N):
            _step(i,self.optimizer,self.loss,self.trainable_variables)
            
    def predict(self):
        if self.use_xy:
            self.y = self.model.predict_y(self.xy_dense)
        else:
            self.y = self.model.predict_y(self.pm_dense.compositions.values[:,[0,1]]/100.0)
        
        y_mean = self.y[0].numpy() 
        self.pm_mean = self.pm_dense.copy(labels=y_mean.argmax(1))
        
        y_var = self.y[1].numpy() 
        self.pm_var = self.pm_dense.copy(labels=y_var.sum(1))
        self.pm_var.plotter.cmap = 'viridis'
            
    @property
    def mean(self):
        return self.pm_mean
    
    @property
    def var(self):
        return self.pm_var
    

    
