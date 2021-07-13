import numpy as np
import pandas as pd
from scipy.stats import norm

class lm0:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        (self.n,self.r) = x.shape
        xx = np.dot(x.T,x)
        xy = np.dot(x.T,y)
        self.xxi = np.linalg.inv(xx)
        self.b = np.linalg.solve(xx,xy)
        e = y - np.dot(x,self.b)
        self.resid = e
        self.vb = self.genvariance(e)
        self.se = np.sqrt(np.diagonal(self.vb))
        self.tstat = np.divide(self.b,self.se)
        self.pval = 2*norm.cdf(-np.abs(self.tstat))
        self.rsq = 1-e.var()/y.var()
        self.adjrsq = 1-(1-self.rsq)*(self.n-1)/(self.n-self.r)
        self.logl = -self.n/2*(np.log(2*np.pi*e.var())+1)
        self.aic = 2*self.r-2*self.logl
        self.bic = np.log(self.n)*self.r-2*self.logl
        nulllike = -self.n/2*(np.log(2*np.pi*y.var())+1)
        self.deviance = 2*(self.logl-nulllike)
    def genvariance(self,e):
        return e.var()*self.xxi
    def predict(self,*args):
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            newx = args[0]
        return np.dot(newx,self.b)
    def tidy(self):
        df = [self.b,self.se,self.tstat,self.pval]
        df = [x.reshape(-1,1) for x in df]
        df = np.hstack(df)
        df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val'])
        return df
    def glance(self):
        df = pd.DataFrame(columns=['r.squared','adj.rsq','r','logl',\
                                   'aic','bic','deviance','df'])
        df.loc[0] = [self.rsq,self.adjrsq,self.r,self.logl,self.aic,\
                     self.bic,self.deviance,self.n-self.r]
        return df

class lm(lm0):
    def __init__(self,x,y):
        (self.n,self.r) = x.shape
        ones = np.ones((self.n,1))
        x = np.hstack((ones,x))
        super(lm,self).__init__(x,y)
    def predict(self,*args):
        if len(args)==1:
            m = args[0].shape
            ones = np.ones((m,1))
            args[0] = np.hstack((ones,args[0]))
        return super(lm,self).predict(*args)

class white(lm):
    def genvariance(self,e):
        meat = np.diagflat(e**2)
        meat = self.x.T.dot(meat).dot(self.x)
        return self.xxi.dot(meat).dot(self.xxi)