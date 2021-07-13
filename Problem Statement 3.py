
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
import random


# ## Question 1

# In[2]:


#creating data generating process
def generate(beta0,beta1):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    x=np.random.multivariate_normal(mean, cov, 1000)
    e= np.random.normal(size=(1000,2))
    y=beta0+beta1*x+e
    return(x,y)


# In[3]:


# running dgp function
x,y=generate(1,3)


# In[4]:


# building linear Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lreg = LinearRegression()


# In[5]:


#splitting Data
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
x, y,random_state=0)


# In[6]:


# splitting data
model = lreg.fit(X_train , y_train)


# In[7]:


model.coef_


# In[8]:


ypred=model.predict(X_test)


# In[9]:


mspesk1 = mean_squared_error(y_test,ypred)
mspesk1


# In[10]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
a=abs(scores.mean())


# In[11]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_test, y_test, cv=10, scoring='neg_mean_squared_error')
u=abs(scores.mean())


# In[12]:


# Changing value of beta0 and beta1 for dgp


# In[13]:


x,y=generate(0.1,1)


# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lreg = LinearRegression()


# In[15]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
x, y,random_state=0)


# In[16]:


# splitting data
model = lreg.fit(X_train , y_train)


# In[17]:


model.coef_


# In[18]:


ypred=model.predict(X_test)


# In[19]:


mspesk2 = mean_squared_error(y_test,ypred)
mspesk2


# In[20]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
m=abs(scores.mean())


# In[21]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_test, y_test, cv=10, scoring='neg_mean_squared_error')
n=abs(scores.mean())


# In[22]:


df={'Beta0':[1,1,0.1,0.1],'beta1':[3,3,1,1],'mspe_actual':[mspesk1,mspesk1,mspesk2,mspesk2],'CV':[5,10,5,10],'mspe_CV':[a,u,m,n],'difference':[(a-mspesk1),(u-mspesk1),(m-mspesk2),(n-mspesk2)]}
df1=pd.DataFrame(data=df)


# In[23]:


print(df1)


# ## From the above dataframe, we can see the difference between actual mspe and cross validation.

# ## Question 2

# In[24]:


#creating data generating process
def generate(beta0,beta1):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    x=np.random.multivariate_normal(mean, cov, 1000)
    e= np.random.normal(size=(1000,2))
    y=beta0+beta1*x+e
    return(x,y)


# In[25]:


x,y=generate(1,3)


# In[26]:


model = smf.OLS(y, x)
model = model.fit()


# In[27]:


r=model.params
r


# In[28]:


## Creating Bootstrap
from sklearn.utils import resample
boot = resample(r, replace=True, n_samples=1000, random_state=1)


# In[29]:


# T-Test Comparision 
stats.ttest_ind(r,boot)


# In[30]:


# Changing value of beta0 and beta1 for dgp
#creating data generating process
def generate(beta0,beta1):
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    x=np.random.multivariate_normal(mean, cov, 1000)
    e= np.random.normal(size=(1000,2))
    y=beta0+beta1*x+e
    return(x,y)


# In[31]:


x,y=generate(0.1,1)


# In[32]:


x= sm.add_constant(x)
model = smf.OLS(y, x)
model = model.fit()


# In[33]:


r=model.params
r


# In[34]:


##Creating Bootstrap
from sklearn.utils import resample
boot = resample(r, replace=True, n_samples=1000, random_state=1)


# In[35]:


# T-Test Comparision 
stats.ttest_ind(r,boot)


# In[36]:


df={'Beta0':[1,0.1],'beta1':[3,1],'No_of_Simulation':[1000,1000],'t-stat':[0.0621,0.01646],'Critical_value':[1.96,1.96],'p-value':[0.95,0.988]}
df1=pd.DataFrame(data=df)


# In[37]:


print(df1)


# ## From the above results, we observe that if we increase the value of beta, then the p- value decreases

# ## Question 3

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


# In[39]:


import pandas as pd
sp500= pd.read_csv('SP500 (3).csv')
gold= pd.read_csv('GOLDAMGBD228NLBM (1).csv')
dcoil= pd.read_csv('DCOILWTICO (1).csv')
bt= pd.read_csv('BTC-USD (2).csv')
dex= pd.read_csv('DEXUSEU (1).csv')


# In[40]:


from IPython.display import display
display(bt.head())
display(sp500.head())
display(gold.head())
display(dex.head())
display(dcoil.head())


# In[41]:


from IPython.display import display
display(bt.shape)
display(sp500.shape)
display(gold.shape)
display(dex.shape)
display(dcoil.shape)


# ## 3.Merging

# In[42]:


df=pd.merge(sp500,gold)
df=pd.merge(df,dcoil)
df=pd.merge(df,dex)
df=pd.merge(df,bt)


# In[43]:


df.head()


# In[44]:


df.shape


# In[45]:


df=df[df['SP500']!='.']
df=df[df['GOLDAMGBD228NLBM']!='.']
df=df[df['DCOILWTICO']!='.']
df=df[df['DEXUSEU']!='.']


# In[46]:


df.head()


# In[47]:


df.info()


# In[48]:


df.shape


# In[49]:


df['GOLDAMGBD228NLBM'] = df['GOLDAMGBD228NLBM'].astype(float)
df['SP500'] = df['SP500'].astype(float)
df['DCOILWTICO'] = df['DCOILWTICO'].astype(float)
df['DEXUSEU'] = df['DEXUSEU'].astype(float)
df['BTC'] = df['BTC'].astype(float)
df['DATE']=pd.to_datetime(df.DATE)
print(df.dtypes)


# In[50]:


c=['BTC', 'DEXUSEU', 'DCOILWTICO', 'SP500', 'GOLDAMGBD228NLBM']
c


# ## 4.Plot

# In[51]:


df[[r for r in c] + 
   ['DATE']].groupby('DATE').sum().plot()


# ## 5. Regression

# In[52]:


import statsmodels.api as sm


# In[53]:


X = np.column_stack((df['SP500'], df['GOLDAMGBD228NLBM'],df['DCOILWTICO'],df['DEXUSEU']))
X = sm.add_constant(X)
model = sm.OLS(df['BTC'],X)
results = model.fit()
print(results.params)


# In[54]:


print(results.summary())


# ## 6.KPSS

# In[55]:


from statsmodels.tsa.api import kpss


# In[56]:


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform  test:
    return(kpss_test(timeseries))


# In[57]:


def kpss_test(times):
    kpsstest=kpss(times,regression='c')
    kpss_output=pd.Series(kpsstest[0:3],index=['Test Statistic','P-value','Lags used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value(%s)'%key] =value
    return kpss_output


# In[58]:


kpss_test(df['BTC']) #not


# In[59]:


kpss_test(df['DEXUSEU']) #not


# In[60]:


kpss_test(df['SP500']) #not


# In[61]:


kpss_test(df['GOLDAMGBD228NLBM']) #not 2.5%


# In[62]:


kpss_test(df['DCOILWTICO']) #not


# In[63]:


# If the test statistic Test Statistic is greater than the critical value of 10,5,2,5,1, we reject the null hypothesis (the sequence is not stationary).
#If the test statistic is less than the critical value, the null hypothesis cannot be rejected (the sequence is stationary).
# For the  data, the value of the test statistic is greater than the critical value in all confidence intervals, so it can be said that the sequence is not stable


# ## 7.Differenced series

# In[64]:


# create a differenced series
def difference(dataset, interval):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# In[65]:


df.head()


# In[66]:


df1=df


# In[67]:


diffsp500=df1['SP500'].diff(23)
plt.plot(diffsp500)
plt.show()


# In[68]:


df1['SP500'] = df1['SP500'] - df1['SP500'].shift(23)
df1['SP500'].dropna().plot()
plt.plot()


# In[69]:


diffbtc=df1['BTC'].diff(23)
plt.plot(diffbtc)
plt.show()


# In[70]:


df1['DEXUSEU'] = df1['DEXUSEU'] - df1['DEXUSEU'].shift(23)
df1['DEXUSEU'].dropna().plot()
plt.plot()


# In[71]:


diffGOLDAMGBD228NLBM=df1['GOLDAMGBD228NLBM'].diff(23)
plt.plot(diffGOLDAMGBD228NLBM)
plt.show()


# In[72]:


df1['GOLDAMGBD228NLBM'] = df1['GOLDAMGBD228NLBM'] - df1['GOLDAMGBD228NLBM'].shift(23)
df1['GOLDAMGBD228NLBM'].dropna().plot()
plt.plot()


# In[73]:


diffDCOILWTICO=df1['DCOILWTICO'].diff(23)
plt.plot(diffDCOILWTICO)
plt.show()


# In[74]:


df['DCOILWTICO'] = df1['DCOILWTICO'] - df1['DCOILWTICO'].shift(23)
df['DCOILWTICO'].dropna().plot()


# In[75]:


df1.head()


# In[76]:


df2=df1.dropna() 
df2.head()


# In[77]:


X2 = np.column_stack((df2['SP500'], df2['GOLDAMGBD228NLBM'],df2['DCOILWTICO'],df2['DEXUSEU']))
X2 = sm.add_constant(X2)
model = sm.OLS(df2['BTC'],X2)
results = model.fit()
print(results.params)


# In[78]:


print(results.summary())


# ## 8.Removing dates before 2017

# In[79]:


import datetime

df_before = datetime.date(2016, 12, 31)
bdate = df[df['DATE'] > pd.to_datetime(df_before)]


# In[80]:


bdate.head()


# ## 9.ACF and PACF

# In[81]:


from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = Series.from_csv('BTC-USD (2).csv', header=0)
plot_acf(series)
pyplot.show()


# In[82]:


from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series1 = bdate.iloc[:,1]
plot_acf(series, lags=50)
pyplot.show()


# In[83]:


from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
series1 = bdate.iloc[:,1]
plot_pacf(series, lags=50)
pyplot.show()


# In[84]:


series2 = df.iloc[:,2]
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series2)
pyplot.show()


# In[85]:


from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(series1)
pyplot.show()


# ## 10.ARIMA

# In[86]:


import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA


# In[87]:


lnprice=np.log(df['BTC'])
lnprice
plt.plot(lnprice)
plt.show()
acf_1 =  acf(lnprice)[1:20]
plt.plot(acf_1)
plt.show()
test_df = pandas.DataFrame([acf_1]).T
test_df.columns = ['Pandas Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
pacf_1 =  pacf(lnprice)[1:20]
plt.plot(pacf_1)
plt.show()
test_df = pandas.DataFrame([pacf_1]).T
test_df.columns = ['Pandas Partial Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
result = ts.adfuller(lnprice, 1)
result


# In[88]:


lnprice_diff=lnprice-lnprice.shift()
diff=lnprice_diff.dropna()
acf_1_diff =  acf(diff)[1:20]
test_df = pandas.DataFrame([acf_1_diff]).T
test_df.columns = ['First Difference Autocorrelation']
test_df.index += 1
test_df.plot(kind='bar')
pacf_1_diff =  pacf(diff)[1:20]
plt.plot(pacf_1_diff)
plt.show()


# In[89]:


price_matrix=lnprice.as_matrix()
model = ARIMA(price_matrix, order=(3,1,0))
model_fit = model.fit(disp=0)


# In[90]:


print(model_fit.summary())


# In[91]:


# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# In[92]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 

X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(3,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predict=%f, expect=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[93]:


X = df.values


# In[94]:


series


# In[95]:


# create a differenced series
import numpy
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())


# In[96]:


# create a differenced series
import numpy
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(5,2,2))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())


# ## 11. Forcecast

# In[97]:


forecast = model_fit.predict()


# In[98]:


forecast


# In[99]:


# Plot histogram
np.histogram(forecast, bins=30, density=True)
plt.hist(forecast, bins='auto') 
plt.show()


# In[100]:


predictions=model_fit.predict(1245, 1275, typ='levels')
predictions


# In[101]:


df.tail()


# In[102]:


price_matrix


# In[103]:


lnprice


# ## 12.Periodogram

# In[104]:


import scipy
f, Pxx_den=scipy.signal.periodogram(series, fs=1.0, window=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1)
plt.semilogy(f, Pxx_den)
plt.show()


# In[105]:


newbitcoin = df[488:843]
newbitcoin


# ## 13

# In[106]:


from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

price = newbitcoin.iloc[:,-1].values
date = newbitcoin.iloc[:,0].values
plot_acf(price, ax=pyplot.gca())
pyplot.show()


# In[107]:


plot_pacf(price, ax=pyplot.gca())
pyplot.show()


# ## 14

# In[108]:


from statsmodels.tsa.arima_model import ARIMA
xarima = ARIMA(price,order=(1,0,0)).fit()


# In[109]:


#from statsmodels.tsa.arima_model import ARMAResults

x= xarima.forecast(steps=30)[0]
x

