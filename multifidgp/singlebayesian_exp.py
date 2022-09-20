__doc__ = """
SingleBayesianExp
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

class SingleBayesianExp:

    eps = 1.e-10   # Cutoff for comparison to zero
    # model_parameters = [s r n]
    
    def __init__(self, XData, KData, model_parameters):
        self.Xdata=XData
        self.Kdata=KData
        self.model_parameters=model_parameters

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
    
    def SingleKrig(self, s, r):
        X = self.Xdata
        y = np.log(self.Kdata)
        mu = np.mean(y)
        
        N = len(X)

        # K matrix
        K = self.k(X, X, self.model_parameters) 
        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        P, L, U = sla.lu(K)

        s = s.reshape(-1)
        x_star = s.reshape([1,len(s)])
        dim = x_star.shape

        psi = self.k(x_star, X, self.model_parameters)
        
        # calculate prediction
        mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
        var_star = self.k(x_star, x_star, self.model_parameters) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
        var_star = abs(np.diag(var_star))

        mean_lognormal = np.exp(mean_star + 0.5*var_star)
        var_lognormal = (np.exp(var_star)-1)*np.exp(2*mean_star+var_star)     
        return mean_lognormal, var_lognormal

    def utility(self, s):
        N=100 # Gaussian samplings
        d=abs(np.random.normal(loc=self.SingleKrig(s, self.model_parameters[1])[0], scale=self.SingleKrig(s, self.model_parameters_H[1])[1], size=N))
        r=abs(np.random.normal(loc=self.model_parameters[1], scale=0.2*self.model_parameters[1], size=N))
        
        s = s.reshape(-1)
        s = s.reshape([1,len(s)])
                                 
        Utility = 0
        for i in range(N):
            prior = 0
            for j in range(N):
                prior = prior + 1/N*(self.likelihood(d[i],self.SingleKrig(s, r[j])[0]))
            Utility = Utility + 1/N*(np.log(self.likelihood(d[i],self.SingleKrig(s, r[i])[0]))-np.log(prior))    
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
            max = smax.reshape(-1)
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
            max = smax.reshape(-1)
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
            max = smax.reshape(-1)
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
        #Result = op.minimize(fun = self.utility, x0 = inis, method = 'SLSQP', jac = self.gradient)#, bounds = bnds)
        print('Powell Optimization details:')
        print(Result)
        s = Result.x
        return s