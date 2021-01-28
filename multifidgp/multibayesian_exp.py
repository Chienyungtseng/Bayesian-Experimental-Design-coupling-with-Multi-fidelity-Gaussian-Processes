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
    
    def __init__(self, XData_H, XData_L, KData_H, KData_L,
                 model_parameters_H, model_parameters_L, rho):
        self.Xdata_H=XData_H
        self.Xdata_L=XData_L
        self.Kdata_H=KData_H
        self.Kdata_L=KData_L
        self.model_parameters_H=model_parameters_H
        self.model_parameters_L=model_parameters_L
        self.rho=rho

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
    
    def likelihood(self, d, G):
        p = np.exp(-0.5*(d-G)**2)
        return p
    
    def MultiKrig(self, s, r_H, r_L):
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = np.log(self.Kdata_L)
        y_H = np.log(self.Kdata_H)
        rho = self.rho
        
        sigma_eps_L = 0
        sigma_eps_H = 0

        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H

        # K matrix
        K_LL = self.k(X_L, X_L, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]]))
        K_LH = rho*self.k(X_L, X_H, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]]))
        K_HL = rho*self.k(X_H, X_L, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]]))
        K_HH = (rho**2)*self.k(X_H, X_H, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]])) + self.k(X_H, X_H, np.array([self.model_parameters_H[0], r_H, self.model_parameters_H[2]]))
        
        K_LL = K_LL + np.eye(N_L)*sigma_eps_L
        K_HH = K_HH + np.eye(N_H)*sigma_eps_H
        
        K = np.concatenate([np.concatenate([K_LL,K_LH], axis=1),np.concatenate([K_HL,K_HH], axis=1)])      
        
        K = K + np.eye(N)*self.eps

        # LU Decomposition
        P, L, U = sla.lu(K)

        y = np.concatenate([y_L, y_H]) 
        s = s.reshape(-1)
        x_star = s.reshape([1,len(s)])

        dim = x_star.shape

        psi1 = rho*self.k(x_star, X_L, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]]))
        psi2 = rho**2*self.k(x_star, X_H, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]])) + self.k(x_star, X_H, np.array([self.model_parameters_H[0], r_H, self.model_parameters_H[2]]))
        psi = np.concatenate([psi1, psi2], axis=1)
        
        # calculate prediction
        mean_star = psi@(la.inv(U)@(la.inv(L)@(P.T@y)))
        var_star = rho**2*self.k(x_star, x_star, np.array([self.model_parameters_L[0], r_L, self.model_parameters_L[2]])) + self.k(x_star, x_star, np.array([self.model_parameters_H[0], r_H, self.model_parameters_H[2]])) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
        var_star = abs(np.diag(var_star))
                    
        mean_lognormal = np.exp(mean_star + 0.5*var_star)
        var_lognormal = (np.exp(var_star)-1)*np.exp(2*mean_star+var_star)
                
        output = np.concatenate([mean_lognormal, var_lognormal])
        return output

    def utility(self, s):
        N=100 # Gaussian samplings
        d=abs(np.random.normal(loc=self.MultiKrig(s, self.model_parameters_H[1], self.model_parameters_L[1])[0], scale=self.MultiKrig(s, self.model_parameters_H[1], self.model_parameters_L[1])[1], size=N))
        r_H=abs(np.random.normal(loc=self.model_parameters_H[1], scale=0.01*self.model_parameters_H[1], size=N))
        r_L=abs(np.random.normal(loc=self.model_parameters_L[1], scale=0.01*self.model_parameters_H[1], size=N))
              
        rho = self.rho
        
        s = s.reshape(-1)
        s = s.reshape([1,len(s)])
                                 
        Utility = 0
        for i in range(N):
            prior = 0
            for j in range(N):
                prior = prior + 1/N*(self.likelihood(d[i],self.MultiKrig(s, r_H[j], r_L[j])[0]))
            Utility = Utility + 1/N*(np.log(self.likelihood(d[i],self.MultiKrig(s, r_H[i], r_L[i])[0]))-np.log(prior))    
        Utility=-Utility
        print("Utility = ", -Utility)
        return Utility
    
    def gradient(self, s):
        dx = 0.01
        dy = 0.01
        dz = 0.01
        if len(s)==1:
            grad = (self.utility(s+dx)-self.utility(s-dx))/(2*dx)
        if len(s)==2:
            grad = np.zeros(2)
            grad[0] = (self.utility(s+np.array([dx,0]))-self.utility(s-np.array([dx,0])))/(2*dx)
            grad[1] = (self.utility(s+np.array([0,dy]))-self.utility(s-np.array([0,dy])))/(2*dy)
        if len(s)==3:
            grad = np.zeros(3)
            grad[0] = (self.utility(s+np.array([dx,0,0]))-self.utility(s-np.array([dx,0,0])))/(2*dx)
            grad[1] = (self.utility(s+np.array([0,dy,0]))-self.utility(s-np.array([0,dy,0])))/(2*dy)
            grad[2] = (self.utility(s+np.array([0,0,dz]))-self.utility(s-np.array([0,0,dz])))/(2*dz)
        return grad
    
    # Find maximum Utility by assigning uniform grid with resolution (res) for the sampling location candidates
    def execute_max(self, bnd, res):
        if len(bnd)==1:
            bndx = bnd
            linx = np.linspace(bndx[0], bndx[1], int(abs(bndx[0]-bndx[1])/res)+1)
            s = np.zeros(len(linx))
            Utility = np.zeros(len(s))
            for i in range(len(linx)):
                s[i] = linx[i]
                Utility[i] = self.utility(s[i])   
            maxU = np.min(Utility)
            smax = s[Utility==maxU]
            smax = smax.reshape(-1)
            print("max Utility = ", -maxU, " s = ", smax)
        elif len(bnd)==2:
            bndx = bnd[0, :]
            bndy = bnd[1, :]
            linx = np.linspace(bndx[0], bndx[1], int(abs(bndx[0]-bndx[1])/res)+1)
            liny = np.linspace(bndy[0], bndy[1], int(abs(bndy[0]-bndy[1])/res)+1)
            s = np.zeros([len(linx)*len(liny), 2])
            Utility = np.zeros(len(s))
            for i in range(len(linx)):
                for j in range(len(liny)):
                    s[i*len(liny)+j, 0] = linx[i]
                    s[i*len(liny)+j, 1] = liny[j]
                    Utility[i*len(liny)+j] = self.utility(s[i*len(liny)+j, :])   
            maxU = np.min(Utility)
            smax = s[Utility==maxU, :]
            smax = smax.reshape(-1)
            print("max Utility = ", -maxU, " s = ", smax)
        elif len(bnd)==3:
            bndx = bnd[0, :]
            bndy = bnd[1, :]
            bndz = bnd[2, :]
            linx = np.linspace(bndx[0], bndx[1], int(abs(bndx[0]-bndx[1])/res)+1)
            liny = np.linspace(bndy[0], bndy[1], int(abs(bndy[0]-bndy[1])/res)+1)
            linz = np.linspace(bndz[0], bndz[1], int(abs(bndz[0]-bndz[1])/res)+1)
            s = np.zeros([len(linx)*len(liny)*len(linz), 3])
            Utility = np.zeros(len(s))
            for i in range(len(linx)):
                for j in range(len(liny)):
                    for k in range(len(linz)):
                        s[i*len(liny)*len(linz)+j*len(linz)+k, 0] = linx[i]
                        s[i*len(liny)*len(linz)+j*len(linz)+k, 1] = liny[j]
                        s[i*len(liny)*len(linz)+j*len(linz)+k, 2] = linz[k]
                        Utility[i*len(liny)*len(linz)+j*len(linz)+k] = self.utility(s[i*len(liny)*len(linz)+j*len(linz)+k, :])   
            maxU = np.min(Utility)
            smax = s[Utility==maxU, :]
            smax = smax.reshape(-1)
            print("max Utility = ", -maxU, " s = ", smax)
        else:
            raise Exception("The input sampling location is not in 1D, 2D, or 3D coordinate!")            
        return smax

    # Find maximum Utility by performing numerical optimization for the sampling location candidates
    def execute_optimization(self, inis):
        if len(inis)==1:
            bnds = ((-30, 5))
        if len(inis)==2:
            bnds = ((-30, 5), (-10, 25))
        if len(inis)==3:
            bnds = ((-30, 5), (-10, 25), (0, 200))
        Result = op.minimize(fun = self.utility, x0 = inis, method = 'Powell', bounds = bnds)
        #Result = op.minimize(fun = self.utility, x0 = inis, method = 'TNC', jac = self.gradient, bounds = bnds)
        print('Powell Optimization details:')
        print(Result)
        s = Result.x
        return s