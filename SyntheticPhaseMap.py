import numpy as np
import pandas as pd

import sasmodels.data
import sasmodels.core
import sasmodels.direct_model
import sasmodels.bumps_model

from AFL import TernaryPhaseMap

class SyntheticTernaryPhaseMap(TernaryPhaseMap):
    def __init__(self,compositions,labels):
        #need a fake measurements array
        measurements = compositions.copy(deep=True)
        measurements.values.fill(1)
        super().__init__(compositions,measurements,labels,metadata=None)
        self.sasmodels = SyntheticSASModels()
        
    def __getitem__(self,composition):
        raise NotImplementedError()
        # composition = self.model.compositions.iloc[index]
        # measurement = self.model.measurements.iloc[index]
        # label = self.model.labels.iloc[index]
        # return (composition,measurement,label)
        
    def add_configuration(self,ABS_fname):
        self.sasmodels.add_configuration(ABS_fname)
        
    def add_sasview_model(self,label,model_name,model_kw):
        self.sasmodels.add_sasview_model(label,model_name,model_kw)
        
    

class SyntheticSASModels:
    def __init__(self):
        self.ABS=[]
        self.sasmodels = {}
        
    def add_configuration(self,ABS_fname):
        '''Read in a ABS file using sasmodels to define an instrument configuration'''
        self.ABS.append(sasmodels.data.load_data(ABS_fname))
        
    def add_sasview_model(self,label,model_name,model_kw):
        ## convert label to ordinal_label
        
        calculators = []
        sasdatas = []
        for sasdata in self.ABS:
            model_info    = sasmodels.core.load_model_info(model_name)
            kernel        = sasmodels.core.build_model(model_info)
            calculator    = sasmodels.direct_model.DirectModel(sasdata,kernel)
            calculators.append(calculator)
            sasdatas.append(sasdata)
            
        self.sasmodels[label] = {
            'name':model_name,
            'kw':model_kw,
            'calculators':calculators,
            'sasdata':sasdatas,
        }
    def generate(self,label,noise=0.05):
        model = self.sasmodels[label]
        kw = model['kw']
        calculators = model['calculators']
        sasdatas = model['sasdata']
        
        I_list = []
        I_noise_list = []
        dI_list = []
        for sasdata,calc in zip(sasdatas,calculators):
            I = calc(**model['kw'])
            
            dI_model = sasdata.dy*np.sqrt(I/sasdata.y)
            mean_var= np.mean(dI_model*dI_model/I)
            # dI = sasdata.dy*np.sqrt(noise*noise/mean_var)
            dI = sasdata.dy*noise/mean_var
            
            I_noise = np.random.normal(loc=I,scale=dI)
            
            I = pd.Series(data=I,index=sasdata.x)
            I_noise = pd.Series(data=I_noise,index=sasdata.x)
            dI = pd.Series(data=dI,index=sasdata.x)
            
            I_list.append(I)
            I_noise_list.append(I_noise)
            dI_list.append(dI)
            
        I = pd.concat(I_list).sort_index()
        I_noise = pd.concat(I_noise_list).sort_index()
        dI = pd.concat(dI_list).sort_index()
        return I,I_noise,dI
            
            
        
    
    