__doc__ = """
MultiBayesianExp
=======

Code by Chien-Yung Tseng, University of Illinois Urbana-Champaign
cytseng2@illinois.edu

Summary
-------
Contains class BayesianExp
Contains PyKrige.variogram_models

References
----------
.. [1] Zhang, J., Zeng, L., Chen, C., Chen, D., & Wu, L. (2015).
Efficient Bayesian experimental design for contaminant source identification.
Water Resources Research, 51(1), 576-598.
.. [2] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
Hydrogeology, (Cambridge University Press, 1997) 272 p.

"""

import numpy as np
import scipy.linalg
import scipy.linalg as sla
import numpy.linalg as la
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.optimize as op
from multifidgp.variogram_models import gaussian_variogram_model
from multifidgp.variogram_models import exponential_variogram_model

class MultiBayesianExp:

    eps = 1.e-10   # Cutoff for comparison to zero
    # model_parameters_L = [sL rL nL]
    # model_parameters_H = [sH rH nH]
    # hyp = [ sigma_eps_L  sigma_eps_H rho]
    # hyp = [logsigma1 logtheta1 logsigma2 logtheta2 rho logsigma_eps_L logsigma_eps_H]
    
    def __init__(self, XData_H, XData_L,
                 model_parameters_H, model_parameters_L, rho, f):
        self.Xdata_H=XData_H
        self.Xdata_L=XData_L
        self.model_parameters_H=model_parameters_H
        self.model_parameters_L=model_parameters_L
        self.rho=rho
        self.f=f

    def k(self, X1, X2, model_parameters):
        """Assembles the kriging matrix."""
        if X1.size==len(X1):
            X1=X1.reshape(len(X1),1)
            X2=X2.reshape(len(X2),1)
        d = cdist(X1, X2, 'euclidean')
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))
        # Assign Exponential variogram model
        K[:n1, :n2] = exponential_variogram_model(model_parameters, d)
        return K

    def utility(self, s):
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        model_parameters_L = self.model_parameters_L
        model_parameters_H = self.model_parameters_H
        
        rho = self.rho
        f = self.f
        
        nH = len(X_H)
        nL = len(X_L)
                
        if f==0:
            nL=nL+1
            if s.shape[0]==2:
                X_L = np.concatenate([X_L, np.array([s])],axis=0)
            elif s.shape[0]==1:
                X_L = np.concatenate([X_L, s],axis=0)
        elif f==1:
            nH=nH+1
            if s.shape[0]==2:
                X_H = np.concatenate([X_H, np.array([s])],axis=0)
            elif s.shape[0]==1:
                X_H = np.concatenate([X_H, s],axis=0)
        X = np.concatenate([X_L, X_H],axis=0)

        Utility = 0
        if X.size==len(X):
            for i in range(nL):
                prior = 1/nL*np.sum(self.k(np.array([X_L[i]]),X_L,model_parameters_L)) + 1/nH*(rho**2*np.sum(self.k(np.array([X_L[i]]),X_H,model_parameters_L)) + np.sum(self.k(np.array([X_L[i]]),X_H,model_parameters_H)))
                Utility = Utility + 1/(nL+nH)*(np.log(self.k(np.array([X_L[i]]),np.array([X_L[i]]),model_parameters_L))-np.log(prior))
            for i in range(nH):
                prior = 1/nL*np.sum(self.k(np.array([X_H[i]]),X_L,model_parameters_L)) + 1/nH*(rho**2*np.sum(self.k(np.array([X_H[i]]),X_H,model_parameters_L)) + np.sum(self.k(np.array([X_H[i]]),X_H,model_parameters_H)))
                Utility = Utility + 1/(nL+nH)*(np.log(rho**2*self.k(np.array([X_H[i]]),np.array([X_H[i]]),model_parameters_L)+self.k(np.array([X_H[i]]),np.array([X_H[i]]),model_parameters_H))-np.log(prior))
        else:
            for i in range(nL):
                prior = 1/nL*np.sum(self.k(np.array([X_L[i,:]]),X_L,model_parameters_L)) + 1/nH*(rho**2*np.sum(self.k(np.array([X_L[i,:]]),X_H,model_parameters_L)) + np.sum(self.k(np.array([X_L[i,:]]),X_H,model_parameters_H)))
                Utility = Utility + 1/(nL+nH)*(np.log(self.k(np.array([X_L[i,:]]),np.array([X_L[i,:]]),model_parameters_L))-np.log(prior))
            for i in range(nH):
                prior = 1/nL*np.sum(self.k(np.array([X_H[i,:]]),X_L,model_parameters_L)) + 1/nH*(rho**2*np.sum(self.k(np.array([X_H[i,:]]),X_H,model_parameters_L)) + np.sum(self.k(np.array([X_H[i,:]]),X_H,model_parameters_H)))
                Utility = Utility + 1/(nL+nH)*(np.log(rho**2*self.k(np.array([X_H[i,:]]),np.array([X_H[i,:]]),model_parameters_L)+self.k(np.array([X_H[i,:]]),np.array([X_H[i,:]]),model_parameters_H))-np.log(prior))
        Utility=-Utility
        #print(Utility,s)
        return Utility
    
    def gradient(self, s):
        dx = 0.01
        dy = 0.01
        dz = 0.01
        if len(s)==1:
            grad = (self.utility(s+dx)-self.utility(s-dx))/(2*dx)
        if len(s)==2:
            #if s.shape[0]==1:
                #s = np.array([s[0,0],s[0,1]])
            grad = np.zeros(2)
            grad[0] = (self.utility(s+np.array([dx,0]))-self.utility(s-np.array([dx,0])))/(2*dx)
            grad[1] = (self.utility(s+np.array([0,dy]))-self.utility(s-np.array([0,dy])))/(2*dy)
        if len(s)==3:
            #if s.shape[0]==1:
                #s = np.array([s[0,0],s[0,1],s[0,2]])
            grad = np.zeros(3)
            grad[0] = (self.utility(s+np.array([dx,0,0]))-self.utility(s-np.array([dx,0,0])))/(2*dx)
            grad[1] = (self.utility(s+np.array([0,dy,0]))-self.utility(s-np.array([0,dy,0])))/(2*dy)
            grad[2] = (self.utility(s+np.array([0,0,dz]))-self.utility(s-np.array([0,0,dz])))/(2*dz)
        return grad

    def utility_new(self, s):
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        model_parameters_L = self.model_parameters_L
        model_parameters_H = self.model_parameters_H
        
        rho = self.rho
        f = self.f
        
        nH = len(X_H)
        nL = len(X_L)
        
        X_L_old = X_L
        X_H_old = X_H
        nH_old = len(X_H_old)
        nL_old = len(X_L_old)
        
        if f==0:
            nL=nL+1
            if s.shape[0]==2:
                X_L = np.concatenate([X_L, np.array([s])],axis=0)
            elif s.shape[0]==1:
                X_L = np.concatenate([X_L, s],axis=0)
        elif f==1:
            nH=nH+1
            if s.shape[0]==2:
                X_H = np.concatenate([X_H, np.array([s])],axis=0)
            elif s.shape[0]==1:
                X_H = np.concatenate([X_H, s],axis=0)
        X = np.concatenate([X_L, X_H],axis=0)

        Utility = 0
        
        if X.size==len(X):
            for i in range(nL_old):
                prior = 1/nL_old*np.sum(self.k(np.array([X_L_old[i]]),X_L_old,model_parameters_L)) + 1/nH_old*(rho**2*np.sum(self.k(np.array([X_L_old[i]]),X_H_old,model_parameters_L)) + np.sum(self.k(np.array([X_L_old[i]]),X_H_old,model_parameters_H)))
                Utility = Utility + 1/(nL_old+nH_old)*(np.log(self.k(np.array([X_L[i]]),np.array([X_L[i]]),model_parameters_L))-np.log(prior))
                #Utility = Utility + 1/(nL_old+nH_old)*(np.log(np.sum(self.k(np.array([X_L[i,:]]),X_L,model_parameters_L)))-np.log(prior))
            for i in range(nH_old):
                prior = 1/nL_old*np.sum(self.k(np.array([X_H_old[i]]),X_L_old,model_parameters_L)) + 1/nH_old*(rho**2*np.sum(self.k(np.array([X_H_old[i]]),X_H_old,model_parameters_L)) + np.sum(self.k(np.array([X_H_old[i]]),X_H_old,model_parameters_H)))
                Utility = Utility + 1/(nL_old+nH_old)*(np.log(rho**2*self.k(np.array([X_H[i]]),np.array([X_H[i]]),model_parameters_L)+self.k(np.array([X_H[i]]),np.array([X_H[i]]),model_parameters_H))-np.log(prior))
                #Utility = Utility + 1/(nL_old+nH_old)*(np.log(np.sum(rho**2*self.k(np.array([X_H[i,:]]),X_H,model_parameters_L)+self.k(np.array([X_H[i,:]]),X_H,model_parameters_H)))-np.log(prior))            
        else:
            for i in range(nL_old):
                prior = 1/nL_old*np.sum(self.k(np.array([X_L_old[i,:]]),X_L_old,model_parameters_L)) + 1/nH_old*(rho**2*np.sum(self.k(np.array([X_L_old[i,:]]),X_H_old,model_parameters_L)) + np.sum(self.k(np.array([X_L_old[i,:]]),X_H_old,model_parameters_H)))
                Utility = Utility + 1/(nL_old+nH_old)*(np.log(self.k(np.array([X_L[i,:]]),np.array([X_L[i,:]]),model_parameters_L))-np.log(prior))
                #Utility = Utility + 1/(nL_old+nH_old)*(np.log(np.sum(self.k(np.array([X_L[i,:]]),X_L,model_parameters_L)))-np.log(prior))
            for i in range(nH_old):
                prior = 1/nL_old*np.sum(self.k(np.array([X_H_old[i,:]]),X_L_old,model_parameters_L)) + 1/nH_old*(rho**2*np.sum(self.k(np.array([X_H_old[i,:]]),X_H_old,model_parameters_L)) + np.sum(self.k(np.array([X_H_old[i,:]]),X_H_old,model_parameters_H)))
                Utility = Utility + 1/(nL_old+nH_old)*(np.log(rho**2*self.k(np.array([X_H[i,:]]),np.array([X_H[i,:]]),model_parameters_L)+self.k(np.array([X_H[i,:]]),np.array([X_H[i,:]]),model_parameters_H))-np.log(prior))
                #Utility = Utility + 1/(nL_old+nH_old)*(np.log(np.sum(rho**2*self.k(np.array([X_H[i,:]]),X_H,model_parameters_L)+self.k(np.array([X_H[i,:]]),X_H,model_parameters_H)))-np.log(prior))
        Utility=-Utility
        #print(Utility,s)
        return Utility
    
    def gradient_new(self, s):
        dx = 0.01
        dy = 0.01
        dz = 0.01
        if len(s)==1:
            grad = (self.utility_new(s+dx)-self.utility_new(s-dx))/(2*dx)
        if len(s)==2:
            #if s.shape[0]==1:
                #s = np.array([s[0,0],s[0,1]])
            grad = np.zeros(2)
            grad[0] = (self.utility_new(s+np.array([dx,0]))-self.utility_new(s-np.array([dx,0])))/(2*dx)
            grad[1] = (self.utility_new(s+np.array([0,dy]))-self.utility_new(s-np.array([0,dy])))/(2*dy)
        if len(s)==3:
            #if s.shape[0]==1:
                #s = np.array([s[0,0],s[0,1],s[0,2]])
            grad = np.zeros(3)
            grad[0] = (self.utility_new(s+np.array([dx,0,0]))-self.utility_new(s-np.array([dx,0,0])))/(2*dx)
            grad[1] = (self.utility_new(s+np.array([0,dy,0]))-self.utility_new(s-np.array([0,dy,0])))/(2*dy)
            grad[2] = (self.utility_new(s+np.array([0,0,dz]))-self.utility_new(s-np.array([0,0,dz])))/(2*dz)
        return grad

    def execute(self, inis):
        if len(inis)==1:
            bnds = ((-40, 10))
        if len(inis)==2:
            bnds = ((-40, 10), (-15, 30))
        if len(inis)==3:
            bnds = ((-40, 10), (-15, 30), (0, 200))
        Result = op.minimize(fun = self.utility_new, x0 = inis, method = 'Powell', bounds = bnds)
        #Result = op.minimize(fun = self.utility_new, x0 = inis, method = 'Newton-CG', jac = self.gradient_new, bounds = bnds)
        print('Powell Optimization details:')
        print(Result)
        s = Result.x
        return s