__doc__ = """
=======

Code by Chien-Yung Tseng, University of Illinois Urbana-Champaign
cytseng2@illinois.edu

-------
Contains class MultiKriging
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

class MultiKriging:

    eps = 1.e-10   # Cutoff for comparison to zero
    # model_parameters_L = [sL rL nL]
    # model_parameters_H = [sH rH nH]
    
    def __init__(self, XData_H, KData_H, XData_L, KData_L,
                 model_parameters_H, model_parameters_L):
        self.Xdata_H=XData_H
        self.Kdata_H=KData_H
        self.Xdata_L=XData_L
        self.Kdata_L=KData_L
        self.model_parameters_H=model_parameters_H
        self.model_parameters_L=model_parameters_L

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
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        y=np.concatenate([y_L, y_H])
        
        sigma_eps_L = hyp[0]
        sigma_eps_H = hyp[1]
        rho = hyp[2]
        
        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H

        K_LL = self.k(X_L, X_L, self.model_parameters_L)
        K_LH = rho*self.k(X_L, X_H, self.model_parameters_L)
        K_HL = rho*self.k(X_H, X_L, self.model_parameters_L)
        K_HH = (rho**2)*self.k(X_H, X_H, self.model_parameters_L) + self.k(X_H, X_H, self.model_parameters_H)
 
        K_LL = K_LL + np.eye(N_L)*sigma_eps_L
        K_HH = K_HH + np.eye(N_H)*sigma_eps_H
 
        K = np.concatenate([np.concatenate([K_LL,K_LH], axis=1),np.concatenate([K_HL,K_HH], axis=1)])
 
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
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        y=np.concatenate([y_L, y_H])
        
        sigma_eps_L = hyp[0]
        sigma_eps_H = hyp[1]
        rho = hyp[2]
        
        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H
 
        K_LL = self.k(X_L, X_L, self.model_parameters_L)
        K_LH = rho*self.k(X_L, X_H, self.model_parameters_L)
        K_HL = rho*self.k(X_H, X_L, self.model_parameters_L)
        K_HH = (rho**2)*self.k(X_H, X_H, self.model_parameters_L) + self.k(X_H, X_H, self.model_parameters_H)
 
        K_LL = K_LL + np.eye(N_L)*sigma_eps_L
        K_HH = K_HH + np.eye(N_H)*sigma_eps_H
 
        K = np.concatenate([np.concatenate([K_LL,K_LH], axis=1),np.concatenate([K_HL,K_HH], axis=1)])

        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        global P, L, U
        P, L, U = sla.lu(K)
        
        alpha = la.inv(U)@(la.inv(L)@(P.T@y))

        # Derivatives
        D_NLML = 0*hyp
        Q = la.inv(U)@(la.inv(L)@(P.T@np.eye(N))) - alpha@alpha.T

        DK_LL = np.zeros([N_L,N_L])
        DK_LH = self.k(X_L, X_H, self.model_parameters_L)
        DK_HL = self.k(X_H, X_L, self.model_parameters_L)
        DK_HH = (2*rho)*self.k(X_H, X_H, self.model_parameters_L)
        DK = np.concatenate([np.concatenate([DK_LL,DK_LH], axis=1),np.concatenate([DK_HL,DK_HH], axis=1)])
        
        D_NLML[2] = sum(sum(Q*DK))/2 # Derivatives for rho
        D_NLML[0] = sigma_eps_L*np.trace(Q[0:N_L,0:N_L])/2  # Derivatives for eps_L
        D_NLML[1] = sigma_eps_H*np.trace(Q[N_L:,N_L:])/2  # Derivatives for eps_H
        return D_NLML
    
    def Hessian(self, hyp):
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        y=np.concatenate([y_L, y_H])
        
        sigma_eps_L = hyp[0]
        sigma_eps_H = hyp[1]
        rho = hyp[2]
        
        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H
 
        K_LL = self.k(X_L, X_L, self.model_parameters_L)
        K_LH = rho*self.k(X_L, X_H, self.model_parameters_L)
        K_HL = rho*self.k(X_H, X_L, self.model_parameters_L)
        K_HH = (rho**2)*self.k(X_H, X_H, self.model_parameters_L) + self.k(X_H, X_H, self.model_parameters_H)
 
        K_LL = K_LL + np.eye(N_L)*sigma_eps_L
        K_HH = K_HH + np.eye(N_H)*sigma_eps_H
 
        K = np.concatenate([np.concatenate([K_LL,K_LH], axis=1),np.concatenate([K_HL,K_HH], axis=1)])
 
        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        global P, L, U
        P, L, U = sla.lu(K)
        
        alpha = la.inv(U)@(la.inv(L)@(P.T@y))

        # Derivatives
        DD_NLML = np.zeros([len(hyp), len(hyp)])
        invK = la.inv(U)@(la.inv(L)@(P.T@np.eye(N)))
        DQ = 0.5*invK@(np.eye(N)-2*y@y.T@invK)@invK

        DK_LL = np.zeros([N_L,N_L])
        DK_LH = self.k(X_L, X_H, self.model_parameters_L)
        DK_HL = self.k(X_H, X_L, self.model_parameters_L)
        DK_HH = (2*rho)*self.k(X_H, X_H, self.model_parameters_L)
        DK = np.concatenate([np.concatenate([DK_LL,DK_LH], axis=1),np.concatenate([DK_HL,DK_HH], axis=1)])
        
        DD_NLML[2, 2] = sum(sum(DQ*DK**2)) # Derivatives for rho^2
        DD_NLML[0, 0] = sigma_eps_L**2*np.trace(DQ[0:N_L,0:N_L])  # Derivatives for eps_L^2
        DD_NLML[1, 1] = sigma_eps_H**2*np.trace(DQ[N_L:,N_L:])  # Derivatives for eps_H^2
        DD_NLML[0, 1] = 0
        DD_NLML[0, 2] = sigma_eps_L*np.trace(DQ[0:N_L,0:N_L]*DK[0:N_L,0:N_L])
        DD_NLML[1, 2] = sigma_eps_H*np.trace(DQ[N_L:,N_L:]*DK[N_L:,N_L:])
        DD_NLML[1, 0] = DD_NLML[0, 1]
        DD_NLML[2, 0] = DD_NLML[0, 2]
        DD_NLML[2, 1] = DD_NLML[1, 2]
        return DD_NLML

    def execute1D(self, xx):
        # initialhyp = [ sigma_eps_L  sigma_eps_H rho]
        inihyp = np.array([0, 0, 0])
        bnds = ((-5, 2), (-5, 2), (0, 10))
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient, bounds = bnds)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        rho = hyp[-1]
        # Set up the limit range of rho and assign the value when reaching the limit
        if rho>1:
            hyp = np.array([0, 0, 0.8])
            rho = hyp[-1]
            self.likelihood(hyp)
        y = np.concatenate([y_L, y_H])
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
            psi1 = rho*self.k(x_star, X_L, self.model_parameters_L)
            psi2 = rho**2*self.k(x_star, X_H, self.model_parameters_L) + self.k(x_star, X_H, self.model_parameters_H)
            psi = np.concatenate([psi1, psi2], axis=1)
        
            # calculate prediction
            mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            var_star = rho**2*self.k(x_star, x_star, self.model_parameters_L) + self.k(x_star, x_star, self.model_parameters_H) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
            var_star = abs(np.diag(var_star))
            
            # Combine mean_star and var_star
            if (i==0):
                mean_star_all = mean_star
                var_star_all = var_star
            else:
                mean_star_all = np.append([mean_star_all], [mean_star])
                var_star_all = np.append([var_star_all], [var_star])
        
        #mean_star_all = mean_star_all.reshape(dim[0])
        #var_star_all = var_star_all.reshape(dim[0])

        return mean_star_all, var_star_all, rho

    def execute2D(self, xx, yy):
        # initialhyp = [ sigma_eps_L  sigma_eps_H rho]
        inihyp = np.array([0, 0, 0])
        bnds = ((-5, 2), (-5, 2), (0, 1))
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient, bounds = bnds)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        rho = hyp[-1]

        y = np.concatenate([y_L, y_H])
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
            psi1 = rho*self.k(x_star, X_L, self.model_parameters_L)
            psi2 = rho**2*self.k(x_star, X_H, self.model_parameters_L) + self.k(x_star, X_H, self.model_parameters_H)
            psi = np.concatenate([psi1, psi2], axis=1)
            # calculate prediction
            mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            var_star = rho**2*self.k(x_star, x_star, self.model_parameters_L) + self.k(x_star, x_star, self.model_parameters_H) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
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
        
        #mean_lognormal = np.exp(mean_star_all + 0.5*var_star_all)
        #var_lognormal = (np.exp(var_star_all)-1)*np.exp(2*mean_star_all+var_star_all)
        
        return mean_star_all, var_star_all, rho
    
    def execute3D(self, xx, yy, zz):
        # initialhyp = [ sigma_eps_L  sigma_eps_H rho]
        inihyp = np.array([0, 0, 0])
        bnds = ((-5, 2), (-5, 2), (0, 10))
        Result = op.minimize(fun = self.likelihood, x0 = inihyp, method = 'TNC', jac = self.Gradient, hess = self.Hessian, bounds = bnds)
        print('TNC Optimization details:')
        print(Result)
        hyp = Result.x
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H
        rho = hyp[-1]
        # Set up the limit range of rho and assign the value when reaching the limit
        if rho>1:
            hyp = np.array([0, 0, 0.8])
            rho = hyp[-1]
            self.likelihood(hyp)
        D = np.size(X_H)/len(X_H)
        y = np.concatenate([y_L, y_H])
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
            psi1 = rho*self.k(x_star, X_L, self.model_parameters_L)
            psi2 = rho**2*self.k(x_star, X_H, self.model_parameters_L) + self.k(x_star, X_H, self.model_parameters_H)
            psi = np.concatenate([psi1, psi2], axis=1)
        
            # calculate prediction
            mean_star = mu + psi@(la.inv(U)@(la.inv(L)@(P.T@(y-mu))))
            var_star = rho**2*self.k(x_star, x_star, self.model_parameters_L) + self.k(x_star, x_star, self.model_parameters_H) - psi@(la.inv(U)@(la.inv(L)@(P.T@psi.T))) 
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
        
        return mean_star_all, var_star_all, rho
    
    def MultiKrig2D(self, xx, yy, r):
        X_L = self.Xdata_L
        X_H = self.Xdata_H
        y_L = self.Kdata_L
        y_H = self.Kdata_H

        sigma_eps_L = 0
        sigma_eps_H = 0
        rho = r

        y = np.concatenate([y_L, y_H])
        mu = np.mean(y)

        N_L = len(X_L)
        N_H = len(X_H)
        N = N_L + N_H
 
        K_LL = self.k(X_L, X_L, self.model_parameters_L)
        K_LH = rho*self.k(X_L, X_H, self.model_parameters_L)
        K_HL = rho*self.k(X_H, X_L, self.model_parameters_L)
        K_HH = (rho**2)*self.k(X_H, X_H, self.model_parameters_L) + self.k(X_H, X_H, self.model_parameters_H)
 
        K_LL = K_LL + np.eye(N_L)*sigma_eps_L
        K_HH = K_HH + np.eye(N_H)*sigma_eps_H
 
        K = np.concatenate([np.concatenate([K_LL,K_LH], axis=1),np.concatenate([K_HL,K_HH], axis=1)])
        K = K + np.eye(N)*self.eps
        
        # LU Decomposition
        PP, LL, UU = sla.lu(K)        
       
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
            psi1 = rho*self.k(x_star, X_L, self.model_parameters_L)
            psi2 = rho**2*self.k(x_star, X_H, self.model_parameters_L) + self.k(x_star, X_H, self.model_parameters_H)
            psi = np.concatenate([psi1, psi2], axis=1)
        
            # calculate prediction
            mean_star = mu + psi@(la.inv(UU)@(la.inv(LL)@(PP.T@(y-mu))))
            var_star = rho**2*self.k(x_star, x_star, self.model_parameters_L) + self.k(x_star, x_star, self.model_parameters_H) - psi@(la.inv(UU)@(la.inv(LL)@(PP.T@psi.T))) 
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
        
        #mean_lognormal = np.exp(mean_star_all + 0.5*var_star_all)
        #var_lognormal = (np.exp(var_star_all)-1)*np.exp(2*mean_star_all+var_star_all)
        
        return mean_star_all, var_star_all
    
    
