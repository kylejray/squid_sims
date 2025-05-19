from re import L
from sympy import *
from scipy.optimize import fsolve
import numpy as np

px, pxdc, beta, dbeta, g = symbols('phi_x phi_xdc beta beta_delta gamma', real=True)
p, pdc = var('phi phi_dc', real=True)


quadratic_p =  .5 * (p-px)**2
quadratic_pdc = .5* g  * (pdc-pxdc)**2
beta_term = beta *cos(.5*pdc)*cos(p)
dbeta_term = -dbeta * sin(.5*pdc) * sin(p)


U =quadratic_p + quadratic_pdc + beta_term + dbeta_term
U_assym = expand(U.subs({beta:0, g:0})-.5*p**2)

def find_px(db, p_dc, mode='min_of_max'):
    if db==0:
        return 0
    if mode=='min_of_max':
        ans_px = nsolve(U_assym.subs(p, acos(px/(dbeta*sin(.5*pdc)))).subs({pdc:p_dc, dbeta:db}), px, .1)
        if np.isclose(float(im(ans_px)),0):
            return float(re(ans_px))
        else:
            return -db*sin(p_dc/2)
    if mode=='min_of_mid':
        return float(-db*sin(p_dc/2))

'''
#params 2:
kT = 6.9E-24
value_dict = {'C':530E-15, 'R':2.1, 'L':140E-12}
Dev2 = DeviceParams(value_dict)
'''

required_attributes = {'C':4E-9, 'R':371, 'L':1E-9, 'alpha':2.07E-15/(2*np.pi)}


class DeviceParams():
    def __init__(self, val_dict={}):
        self.meta_params = ['gamma', 'beta','dbeta', 'U_0']
        self.dev_params = ['ell', 'I_plus', 'I_minus', 'L']

        for dictionary in [required_attributes, val_dict]:
            for key, value in dictionary.items():
                setattr(self, key, value)
        
        init_dict={}
        if not hasattr(self, 'ell'):
                init_dict['gamma'] = 12
        if not hasattr(self, 'I_plus'):
                init_dict['beta'] = 6.2
        if not hasattr(self, 'I_minus'):
                init_dict['dbeta']= .1
        self.change_vals(init_dict)

        if not hasattr(self, 'kT_prime'):
                self.kT_prime = .41*1.38E-23/self.U_0
        
        
    def refresh(self):
        self.gamma = self.L / (2*self.ell)
        self.beta = self.I_plus * self.L / self.alpha
        self.dbeta = self.I_minus * self.L / self.alpha
        self.U_0 = self.alpha**2 / self.L
    
    def meta_refresh(self):
        self.ell = self.L/(2*self.gamma)
        self.I_plus = self.beta * self.alpha / self.L
        self.I_minus = self.dbeta * self.alpha / self.L
        self.L = self.alpha**2 / self.U_0
        

    def change_vals(self, val_dict):
        assert all( not((p[0] in val_dict.keys()) and (p[1] in val_dict.keys())) for p in zip(self.dev_params,self.meta_params)), 'tried to change param and its meta param at the same time, for example gamma and ell'

        if any(key in self.meta_params for key in val_dict.keys()):
            meta_dict = {key:value for key,value in val_dict.items() if key in self.meta_params}
            val_dict = {key:value for key,value in val_dict.items() if key not in self.meta_params}

        self.U_0 = self.alpha**2 / self.L

        try:
            for key, value in val_dict.items():
                setattr(self, key, value)
            self.refresh()
        except: pass

        try:
            for key,value in meta_dict.items():
                setattr(self, key, value)
            self.meta_refresh()
        except: pass

    def quantum_ratio(self):
        return 1.054E-34 / (np.sqrt(self.L*self.C)*self.kT_prime*self.U_0)
    
    def to_dict(self):
        output_dict= { key:value for key, value in self.__dict__.items() }
        output_dict['quantum'] = self.quantum_ratio()
        for item in ['dev_params', 'meta_params']:
            del(output_dict[str(item)])
        return output_dict

    def get_temp(self):
        return self.kT_prime * self.U_0 / 1.38E-23

    
    
    def perturb(self, bias=0, spread=.025, params=['C','R','L', 'ell', 'I_plus', 'I_minus']):
        new_vals = []
        for param in params:
            curr_val = self.__dict__[param]
            new_val = curr_val * (1+np.random.normal(bias, spread))
            new_vals.append(new_val)
        self.change_vals(dict(zip(params,new_vals)))

round_vals = {'C':4E-9, 'R':4E2, 'L':5E-10, 'alpha':2.07E-15/(2*np.pi), 'kT_prime':.05}

RoundDevice = DeviceParams(val_dict=round_vals)
RoundDevice.change_vals({'beta':6})   



def fidelity(jumps):
    
    out = {}
    names = ['+ to -', '- to +']
    tot_fails = 0
    total =0


    for i, key in enumerate(jumps):
        length = len(jumps[key])
        succ, tot = sum(jumps[key]==2), sum(jumps[key]!=0)
        out[names[i]] = succ/tot
        tot_fails += tot-succ
        total += tot
    if total != length:
        print ('missed counting {} trajectories in fidelity'.format(length-total))
    out['overall'] = 1-tot_fails/total

    return out

def fidelity_array(jump_array):

    _, _, N = jump_array.shape

    out = {}
    names = ['+ to -', '- to +']

    succ = np.sum(jump_array==2, axis=-1)
    total = np.sum(jump_array!=0, axis=-1)
    tot_diff = np.sum(total, axis=-1)-N

    if sum(tot_diff!=0) > 0:
        print ('missed counting {} trajectories in fidelity in {} trials'.format(sum(tot_diff), sum(tot_diff!=0)))

    for i, name in enumerate(names):
        out[name] = succ[:,i]/total[:,i]
    
    out['overall'] =1- (total-succ).sum(axis=-1)/total.sum(axis=-1)

    return out



    



             
        
        


