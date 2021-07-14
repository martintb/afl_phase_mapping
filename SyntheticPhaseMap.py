import numpy as np
import pandas as pd
import pickle

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
        
    def save(self,fname):
        sasmodels = {'ABS_fname':[f.filename for f in self.sasmodels.ABS]}
        sasmodels['models'] = {}
        for label,values in self.sasmodels.sasmodels.items():
            sasmodels['models'][label] = {
                'model_name':values['name'],
                'model_kw':values['kw'],
            }
            
        out_dict = {}
        out_dict['compositions'] = self.model.compositions
        out_dict['labels'] = self.model.labels
        out_dict['sasmodels'] = sasmodels
        
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
            labels       = in_dict['labels'],
        )
        
        for fname in in_dict['sasmodels']['ABS_fname']:
            pm.add_configuration(fname)
            
        for label,model in in_dict['sasmodels']['models'].items():
            pm.add_sasview_model(label,model['model_name'],model['model_kw'])
            
        return pm
        
    def add_configuration(self,ABS_fname):
        self.sasmodels.add_configuration(ABS_fname)
        
    def add_sasview_model(self,label,model_name,model_kw):
        self.sasmodels.add_sasview_model(label,model_name,model_kw)
        
    def measure(self,composition,noise=0.05):
        phases = self.locate(composition)
        if len(phases)==0:
            label = 'D'
        elif len(phases)==1:
            label = phases[0]
        else:
            label = phases[0]
        
        _,measurement,_ = self.sasmodels.generate(label,noise=noise)
        return label,measurement
    

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
            
            
        
    
    