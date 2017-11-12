import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
%matplotlib inline

#read file#
file = r'E:\tc_df.csv'
df = pd.read_csv(file, header=0, index_col=0)
df.index = pd.to_datetime(df.index)

share = pd.read_csv(r'E:\mfd_day_share_interest.csv',header=0,index_col=None,usecols=[0,1])
shibor = pd.read_csv(r'E:\mfd_bank_shibor.csv',header=0,index_col=None,usecols=[0,1])

def to_datetime(dt_time):
    year = int(dt_time/10000)
    month = int(dt_time/100)-int(dt_time/10000)*100
    day = dt_time-int(dt_time/100)*100
    return pd.datetime(year, month, day)

share['mfd_date'] = share['mfd_date'].apply(to_datetime)
share.set_index('mfd_date', inplace=True)
df = pd.merge(df, share, how='left', left_index=True, right_index=True)

shibor['mfd_date'] = shibor['mfd_date'].apply(to_datetime)
shibor.set_index('mfd_date', inplace=True)
df = pd.merge(df, shibor, how='left', left_index=True, right_index=True)

#FFT transform(查看显著周期)#
from scipy.fftpack import fft
def plot_with_fft(xdata):    
    days = [r for r in range(xdata.shape[0])]
    fig = plt.figure(1,figsize=[15,5])
    fft_complex = fft(xdata)
    fft_mag = [np.sqrt(np.real(x)*np.real(x)+np.imag(x)*np.imag(x)) for x in fft_complex]
    fft_xvals = [day / days[-1] for day in days]
    npts = len(fft_xvals) // 2 + 1
    fft_mag = fft_mag[:npts]
    fft_xvals = fft_xvals[:npts]
        
    plt.ylabel('FFT Magnitude')
    plt.xlabel(r"Frequency [days]$^{-1}$")
    plt.title('Fourier Transform')
    plt.plot(fft_xvals[1:],fft_mag[1:])
    # Draw lines at 1, 1/2, and 1/3 week periods
    plt.axvline(x=1./7,color='blue',alpha=0.3)
    plt.axvline(x=2./7,color='red',alpha=0.3)
    plt.show()

#total_purchase_amt#
#plot the data
df['total_purchase_amt'].plot()
plot_with_fft(df['total_purchase_amt'])
xdata = df['total_purchase_amt']

#prophet predict
ts = pd.DataFrame()
ts['ds'] = xdata.index
ts['y'] = xdata.values
ts['y'] = np.log(ts['y'])

hld_3d = pd.DataFrame({
  'holiday': 'hld_3d',
  'ds': pd.to_datetime(['2013-09-19', '2014-01-01', '2014-04-05',
                        '2014-05-01', '2014-05-31', '2014-09-06', '2014-10-01']),
  'lower_window': -3,
  'upper_window': 4,
})

hld_7d = pd.DataFrame({
  'holiday': 'hld_7d',
  'ds': pd.to_datetime(['2013-10-01', '2014-01-31']),
  'lower_window': -3,
  'upper_window': 8,
})
holidays = pd.concat((hld_3d, hld_7d))

m = Prophet(holidays=holidays, holidays_prior_scale=5)
m.add_seasonality(name='half-weekly', period=3.5, fourier_order=2, prior_scale=5)

m.fit(ts)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

m.plot(forecast)
m.plot_components(forecast)

#check the fitting results
plt.ylabel('purchase')
plt.xlabel('Day')

plt.plot(ts['ds'],xdata,label = 'true' )
plt.plot(ts['ds'],np.exp(forecast.iloc[:-30,-1].values),label = 'predict' )

#save purchase predict values
purchase = np.exp(forecast.iloc[-30:,-1].values)

#total_redeem_amt#
#plot the data
df['total_redeem_amt'].plot()
plot_with_fft(df['total_redeem_amt'])

#redeem prophet predict
xdata = df['total_redeem_amt']

ts = pd.DataFrame()
ts['ds'] = xdata.index
ts['y'] = xdata.values
ts['y'] = np.log(ts['y'])

hld_3d = pd.DataFrame({
  'holiday': 'hld_3d',
  'ds': pd.to_datetime(['2013-09-19', '2014-01-01', '2014-04-05',
                        '2014-05-01', '2014-05-31', '2014-09-06', '2014-10-01']),
  'lower_window': -2,
  'upper_window': 4,
})

hld_7d = pd.DataFrame({
  'holiday': 'hld_7d',
  'ds': pd.to_datetime(['2013-10-01', '2014-01-31']),
  'lower_window': -2,
  'upper_window': 8,
})
holidays = pd.concat((hld_3d, hld_7d))

def peakpoint(ds):
    date = pd.to_datetime(ds)
    if date>pd.datetime(2014,6,13) and  date<pd.datetime(2014,7,10):
        return 1
    else:
        return 0
ts['peakpoint'] = ts['ds'].apply(peakpoint)

m = Prophet(holidays=holidays, holidays_prior_scale=5)
m.add_seasonality(name='half-weekly', period=3.5, fourier_order=2, prior_scale=5)
m.add_regressor('peakpoint',prior_scale=10)

m.fit(ts)
future = m.make_future_dataframe(periods=30)
future['peakpoint'] = future['ds'].apply(peakpoint)
forecast = m.predict(future)

m.plot(forecast)
m.plot_components(forecast)

#check the fitting results
plt.ylabel('redeem')
plt.xlabel('Day')

#save redeem predict values
redeem = np.exp(forecast.iloc[-30:,-1].values)

#deal with the data format#
submit = pd.DataFrame()
submit['report_date'] = pd.date_range('2014/09/01','2014/09/30') 
submit['purchase'] = purchase
submit['redeem'] = redeem

def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

submit['report_date'] = submit['report_date'].apply(to_integer) 
submit['purchase'] = submit['purchase'].astype('int64')
submit['redeem'] = submit['redeem'].astype('int64')

#write to file#
submit.to_csv(r'E:\tc_comp_predict_table.csv', header=False, index=False)
plt.plot(ts['ds'],xdata,label = 'true' )
plt.plot(ts['ds'],np.exp(forecast.iloc[:-30,-1].values),label = 'predict' )
