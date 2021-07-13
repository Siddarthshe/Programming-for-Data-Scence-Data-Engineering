
# coding: utf-8

# In[112]:


import numpy as np
import pandas as pd
from scipy.stats import norm

Flogit = lambda x: 1/(1+np.exp(-x))
flogit = lambda x: Flogit(x)*(1-Flogit(x))
def NewtonRhapson(xinit,J,H):
    x = xinit
    for i in range(1000):
        #upd = np.linalg.inv(H(x)).dot(J(x))
        upd = np.linalg.solve(H(x),J(x))
        x -= upd
        if (np.power(upd,2)).sum()<1e-16: return(x,J(x),H(x),i)
    raise Exception('did not converge')

    
def kfold(model,stat,x,y,k):
    n = y.shape[0]
    perm = np.random.permutation(n)
    siz = n//k
    mod = model(x,y)
    r = stat(mod,x,y).size
    outp = np.zeros((r,1))
    for i in range(k):
        test = perm[siz*i:siz*(i+1)]
        trainl = perm[:siz*i]
        trainu = perm[siz*(i+1):]
        train = np.hstack((trainl,trainu))
        mod = model(x[train,:],y[train])
        outp += stat(mod,x[test,:],y[test])
    return outp/k

class logit:
    def __init__(self,x,y):
        n = y.shape[0]
        ones = np.ones((n,1))
        x = np.hstack((ones,x))
        r = x.shape[1]
        (self.n,self.r) = (n,r)
        self.y = y
        self.x = x
        def multifunction(b):
            logL = 0
            dlogL = 0
            ddlogL = 0
            for i in range(n):
                xcur = x[i,:].reshape(-1,1)
                inner = xcur.T.dot(b)
                Fx = Flogit(inner)
                logL += y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
                dlogL += (y[i]-Fx)*xcur
                ddlogL -= flogit(inner)*(xcur.dot(xcur.T))
            return(logL, dlogL, ddlogL)
        b = np.zeros((r,1))
        jac = lambda x: multifunction(x)[1]
        hess = lambda x: multifunction(x)[2]
        (b,J,H,it) = NewtonRhapson(b,jac,hess)
        self.b = b.reshape(-1,1)
        self.vb = -np.linalg.inv(H)
        e = y.reshape(-1,1) - Flogit(x.dot(b)).reshape(-1,1)
        self.resid = e
        self.se = np.sqrt(np.diagonal(self.vb)).reshape(-1,1)
        self.tstat = np.divide(self.b,self.se)
        self.pval = 2*norm.cdf(-np.abs(self.tstat))
        self.logl = multifunction(b)[0][0,0]
        self.aic = 2*self.r-2*self.logl
        self.bic = np.log(self.n)*self.r-2*self.logl
        def multifunction(b):
            logL = 0
            dlogL = 0
            ddlogL = 0
            for i in range(n):
                xcur = np.ones((1,1))
                inner = xcur.T.dot(b)
                Fx = Flogit(inner)
                logL += y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
                dlogL += (y[i]-Fx)*xcur
                ddlogL -= flogit(inner)*(xcur.dot(xcur.T))
            return(logL, dlogL, ddlogL)
        b = np.zeros((1,1))
        (b,J,H,it) = NewtonRhapson(b,jac,hess)
        self.nulllike = multifunction(b)[0][0,0]
        self.deviance = 2*(self.logl-self.nulllike)
        self.mcfrsq = 1-self.logl/self.nulllike
        Fhat = Flogit(x.dot(self.b))
        self.blrsq = 0
        self.vzrsq = 0
        self.efrsq = 0
        self.mzrsq = 0
    def predict(self,*args):
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            m = args[0].shape[0]
            ones = np.ones((m,1))
            newx = np.hstack((ones,args[0]))
        return Flogit(np.dot(newx,self.b))
    def tidy(self):
        df = [self.b,self.se,self.tstat,self.pval]
        df = [x.reshape(-1,1) for x in df]
        df = np.hstack(df)
        df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val'])
        return df
    def glance(self):
        df = pd.DataFrame(columns=['mcfadden.rsq','r','logl',                                   'aic','bic','deviance','df',                                   'bl.rsq','vz.rsq','ef.rsq','mz.rsq'])
        df.loc[0] = [self.mcfrsq,self.r,self.logl,self.aic,                     self.bic,self.deviance,self.n-self.r,                    self.blrsq,self.vzrsq,self.efrsq,self.mzrsq]
        return df
    def mspe(self,xtest,ytest):
        err = ytest - self.predict(xtest)
        return np.array((err**2).mean())

class logitclassifier:
    def __init__(self,x,y):
        n = y.shape[0]
        ones = np.ones((n,1))
        x = np.hstack((ones,x))
        r = x.shape[1]
        (self.n,self.r) = (n,r)
        self.y = y
        self.x = x
        def multifunction(b):
            logL = 0
            dlogL = 0
            ddlogL = 0
            for i in range(n):
                xcur = x[i,:].reshape(-1,1)
                inner = xcur.T.dot(b)
                Fx = Flogit(inner)
                logL += y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
                dlogL += (y[i]-Fx)*xcur
                ddlogL -= flogit(inner)*(xcur.dot(xcur.T))
            return(logL, dlogL, ddlogL)
        b = np.zeros((r,1))
        jac = lambda x: multifunction(x)[1]
        hess = lambda x: multifunction(x)[2]
        (b,J,H,it) = NewtonRhapson(b,jac,hess)
        self.b = b.reshape(-1,1)
        self.vb = -np.linalg.inv(H)
        e = y.reshape(-1,1) - Flogit(x.dot(b)).reshape(-1,1)
        self.resid = e
        self.se = np.sqrt(np.diagonal(self.vb)).reshape(-1,1)
        self.tstat = np.divide(self.b,self.se)
        self.pval = 2*norm.cdf(-np.abs(self.tstat))
        self.logl = multifunction(b)[0][0,0]
        self.aic = 2*self.r-2*self.logl
        self.bic = np.log(self.n)*self.r-2*self.logl
        def multifunction(b):
            logL = 0
            dlogL = 0
            ddlogL = 0
            for i in range(n):
                xcur = np.ones((1,1))
                inner = xcur.T.dot(b)
                Fx = Flogit(inner)
                logL += y[i]*np.log(Fx)+(1-y[i])*np.log(1-Fx)
                dlogL += (y[i]-Fx)*xcur
                ddlogL -= flogit(inner)*(xcur.dot(xcur.T))
            return(logL, dlogL, ddlogL)
        b = np.zeros((1,1))
        (b,J,H,it) = NewtonRhapson(b,jac,hess)
        self.nulllike = multifunction(b)[0][0,0]
        self.deviance = 2*(self.logl-self.nulllike)
        self.mcfrsq = 1-self.logl/self.nulllike
        Fhat = Flogit(x.dot(self.b))
        self.blrsq = 0
        self.vzrsq = 0
        self.efrsq = 1-((np.sum((self.y-Fhat.reshape(1,n))*2))/np.sum((self.y-self.y.mean())*2))
        self.mzrsq = 0
    def tidy(self):
        df = [self.b,self.se,self.tstat,self.pval]
        df = [x.reshape(-1,1) for x in df]
        df = np.hstack(df)
        df = pd.DataFrame(df,columns=['est','std.err','t.stat','p.val'])
        return df
    def glance(self):
        df = pd.DataFrame(columns=['mcfadden.rsq','r','logl',                                   'aic','bic','deviance','df',                                   'bl.rsq','vz.rsq','ef.rsq','mz.rsq'])
        df.loc[0] = [self.mcfrsq,self.r,self.logl,self.aic,                     self.bic,self.deviance,self.n-self.r,                    self.blrsq,self.vzrsq,self.efrsq,self.mzrsq]
        return df
    def predict(self,*args,prob=0.5):
        ans_pred=np.ones((self.x.shape[0]))
        index=0
        predict_class=0
        if len(args)>=2:
            raise Exception('Predict takes 0 or 1 argument')
        elif len(args)==0:
            newx = self.x
        else:
            m = args[0].shape[0]
            ones = np.ones((m,1))
            newx = np.hstack((ones,args[0]))
        ans_pred=np.ones((newx.shape[0],1))
        for i in Flogit(np.dot(newx,self.b)):
            if i>=0.5:
                predict_class=1
                ans_pred[index]=predict_class
            else:
                predict_class=0
                ans_pred[index]=predict_class
            index+=1
        return ans_pred
    def mspe(self,xtest,ytest):
        err = ytest - self.predict(xtest)
        return np.array((err**2).mean())
    def accuracy(self,xtest,ytest):
        length=len(xtest)
        _y=ytest.reshape(-1, 1)
        correct=(_y==self.predict(xtest))
        my_accuracy = (np.sum(correct) / length)
        return my_accuracy
    def F1(self,xtest,ytest):
        
        return True
  
        
        
    


# In[16]:


bwght = pd.read_csv('BWGHT.csv')
bwght['smokes'] = (bwght['cigs']>0).astype(int)


# In[17]:


y = bwght['smokes'].values
x = (bwght['faminc'],bwght['white'])
x = pd.concat(x,1)
x = x.values
logit(x,y).glance()


# In[100]:


np.random.seed(75080)
x = np.random.normal(size=(1000,1))
e = np.random.normal(size=(1000,1))
y = ((2*x+e)>0).astype(int)


# In[106]:


np.random.seed(75080)
kfold(logitclassifier,logitclassifier.mspe,x,y,10)


# In[113]:


np.random.seed(75080)
kfold(logitclassifier,logitclassifier.accuracy,x,y,10)


# In[9]:


np.random.seed(75080)
kfold(logitclassifier,logitclassifier.F1,x,y,10)

