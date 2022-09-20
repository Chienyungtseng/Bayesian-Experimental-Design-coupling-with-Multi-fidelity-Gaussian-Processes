__doc__ = """
=======

Code by Chien-Yung Tseng, University of Illinois Urbana-Champaign
cytseng2@illinois.edu

-------
Contains class SingleKriging
Contains PyKrige.variogram_models

References
----------
.. [1] Raissi, M., & Karniadakis, G. (2016). Deep multi-fidelity 
Gaussian processes. arXiv preprint arXiv:1604.07484.
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

P = 0
L = 0
U = 0

class SingleKriging:

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
        # Assign Gaussian variogram model
        #K[:n1, :n2] = gaussian_variogram_model(model_parameters, d)
        return K

    def likelihood(self, hyp):
        X = self.Xdata
        y = self.Kdata
        
        sigma_eps = hyp
        
        N = len(X)
        D = np.size(X)/len(X)
        print(X)
        print(N)
        K = self.k(X, X, self.model_parameters) 
        K = K + np.eye(N)*sigma_eps
        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        global P, L, U
        P, L, U = sla.lu(K)

        alpha = la.inv(U)@(la.inv(L)@(P.T@y))
        for m in range(40):
            if (np.prod(np.diag(U*np.exp(1)**m))!=0):
                break
        NLML = 0.5*y.T@alpha + 0.5*(np.log(abs(np.prod(np.diag(U*np.exp(1)**m))))-m*N) + np.log(2*np.pi)*N/2
        print(NLML, hyp)
        return NLML
    
    def Gradient(self, hyp):
        X = self.Xdata
        y = self.Kdata
        
        sigma_eps = hyp
        
        N = len(X)
        D = np.size(X)/len(X)

        K = self.k(X, X, self.model_parameters) 
        K = K + np.eye(N)*sigma_eps
        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        global P, L, U
        P, L, U = sla.lu(K)
        
        alpha = la.inv(U)@(la.inv(L)@(P.T@y))

        # Derivatives
        D_NLML = 0*hyp
        Q = la.inv(U)@(la.inv(L)@(P.T@np.eye(N))) - alpha@alpha.T # dL/dK
        D_NLML = sigma_eps*np.trace(Q)/2  # dL/dK*dK/dtheta
        return D_NLML
    
    def execute1D(self, xx):
        # initialhyp = [sigma_eps]
        inihyp = np.array([0])
        bnds = ((-5, 2),)
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient, bounds = bnds)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X = self.Xdata
        y = self.Kdata
        D = np.size(X)/len(X)
        mu = np.mean(y)
        
        dim = xx.shape
        
        xx = xx.reshape(np.size(xx),-1)
        x_star_all = xx
        
        # divided matrix calculation
        for i in range(100):
            if (i==99):
                x_star = x_star_all[int(len(x_star_all)/100)*i:,:]
            else:
                x_star = x_star_all[int(len(x_star_all)/100)*i:int(len(x_star_all)/100)*(i+1),:]
            psi = self.k(x_star, X, self.model_parameters)
        
            # calculate prediction
            mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            var_star = self.k(x_star, x_star, self.model_parameters) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
            var_star = abs(np.diag(var_star))
            
            # Combine mean_star and var_star
            if (i==0):
                mean_star_all = mean_star
                var_star_all = var_star
            else:
                mean_star_all = np.append([mean_star_all], [mean_star])
                var_star_all = np.append([var_star_all], [var_star])
        
        mean_star_all = mean_star_all.reshape(dim[0])
        var_star_all = var_star_all.reshape(dim[0])

        return mean_star_all, var_star_all

    def execute2D(self, xx, yy):
        # initialhyp = [sigma_eps]
        inihyp = np.array([0])
        bnds = ((-5, 2),)
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X = self.Xdata
        y = self.Kdata
        D = np.size(X)/len(X)
        mu = np.mean(y)
        
        dim = xx.shape
        
        xx = xx.reshape(np.size(xx),-1)
        yy = yy.reshape(np.size(yy),-1)
        x_star_all = np.concatenate([xx, yy],axis=1)
        
        # divided matrix calculation
        for i in range(100):
            if (i==99):
                x_star = x_star_all[int(len(x_star_all)/100)*i:,:]
            else:
                x_star = x_star_all[int(len(x_star_all)/100)*i:int(len(x_star_all)/100)*(i+1),:]
            psi = self.k(x_star, X, self.model_parameters)
        
            # calculate prediction
            #mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            mean_star = psi@(la.inv(U)@(la.inv(L)@(P.T@y)))
            var_star = self.k(x_star, x_star, self.model_parameters) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
            var_star = abs(np.diag(var_star))

            # Combine mean_star and var_star
            if (i==0):
                mean_star_all = mean_star
                var_star_all = var_star
            else:
                mean_star_all = np.append([mean_star_all], [mean_star])
                var_star_all = np.append([var_star_all], [var_star])
        
        mean_star_all = mean_star_all.reshape(dim[0], dim[1])
        var_star_all = var_star_all.reshape(dim[0], dim[1])
        
        return mean_star_all, var_star_all
    
    def execute3D(self, xx, yy, zz):
        # initialhyp = [sigma_eps]
        inihyp = np.array([0])
        bnds = ((-5, 2),)
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient, hess = self.Hessian, bounds = bnds)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X = self.Xdata
        y = self.Kdata
        D = np.size(X)/len(X)
        mu = np.mean(y)
        
        dim = xx.shape
        
        xx = xx.reshape(np.size(xx),-1)
        yy = yy.reshape(np.size(yy),-1)
        zz = zz.reshape(np.size(zz),-1)
        x_star_all = np.concatenate([xx, yy, zz],axis=1)
        
        # divided matrix calculation
        for i in range(100):
            if (i==99):
                x_star = x_star_all[int(len(x_star_all)/100)*i:,:]
            else:
                x_star = x_star_all[int(len(x_star_all)/100)*i:int(len(x_star_all)/100)*(i+1),:]
            psi = self.k(x_star, X, self.model_parameters)
        
            # calculate prediction
            mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            var_star = self.k(x_star, x_star, self.model_parameters) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
            var_star = abs(np.diag(var_star))
            
            # Combine mean_star and var_star
            if (i==0):
                mean_star_all = mean_star
                var_star_all = var_star
            else:
                mean_star_all = np.append([mean_star_all], [mean_star])
                var_star_all = np.append([var_star_all], [var_star])
        
        mean_star_all = mean_star_all.reshape(dim[0], dim[1], dim[2])
        var_star_all = var_star_all.reshape(dim[0], dim[1], dim[2])
        
        return mean_star_all, var_star_all
