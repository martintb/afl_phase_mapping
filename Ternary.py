import numpy as np

class Ternary:
    def comp2cart(self,compositions):
        try:
            t = compositions.values.copy()
        except AttributeError:
            t = compositions.copy()
            
        # Convert ternary data to cartesian coordinates.
        xy = np.zeros((t.shape[0],2))
        xy[:,1] = t[:,1]*np.sin(60.*np.pi / 180.) / 100.
        xy[:,0] = t[:,0]/100. + xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
        return xy