#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:32:21 2023

@author: alli
"""

#%%
import os
import numpy as np
import pandas as pd

from scipy import signal as sig
import matplotlib
import matplotlib.pyplot as plt
import netCDF4 as nc
#%%
ddir = os.path.join('.','precipitation')
print(ddir)
print()
pr = nc.MFDataset(f'{ddir}/*.nc')
print(f'Variables: {pr.variables}')
print()
print(f'Dimensions: {pr.dimensions}')

#%%
print(f'Variable Keys: {pr.variables.keys()}')

#%%
print(pr.variables['time'][1])

#%%
# unit of time: days
# start time is 1901-01-01 00:00:00

# getting the day after 2953 days
from datetime import datetime, timedelta
first_day = "1901-01-01"
first_day_converted = datetime.strptime(first_day,'%Y-%m-%d')
print(first_day_converted)
new_day = first_day_converted + timedelta(days=2953)
print(new_day)

#%%

ddir = os.path.join('.','temperature')
print(ddir)
print()
tasmax = nc.MFDataset(f'{ddir}/*.nc')
print(f'Variables: {tasmax.variables}')
print()
print(f'Dimensions: {tasmax.dimensions}')


#%%

for i in range(len(pr.variables['lat'])):
    for j in range(len(pr.variables['lon'])):
        if pr.variables['lat'][i] == 31.25 and pr.variables['lon'][j] == 76.25:
            lat, lon = i,j
            break

print(lat, lon)

#%%

df = pd.DataFrame(columns=['time','pr', 'tasmax'])

for i in range(len(pr.variables['pr'])):
    df.loc[i] = np.ma.getdata(pr.variables['time'][i]), np.ma.getdata(pr.variables['pr'][i][lat][lon]), np.ma.getdata(tasmax.variables['tasmax'][i][lat][lon])

print(df)

#%%
df["time_converted"] = ''
start_date = "1901-01-01"
start_date = datetime.strptime(start_date,'%Y-%m-%d')
for i in range(len(df["time"])):
    no_days = int(df["time"].loc[i])
    df["time_converted"].loc[i] = start_date + timedelta(days=no_days)

print(df)

#%%

plt.plot(df['time_converted'], df['pr'], color='green')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Precipitation Against Time')
plt.show()

#%%

plt.plot(df['time_converted'], df['tasmax'], color='red')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Against Time')
plt.show()


#%%

ft = np.fft.fft(df["pr"])

df["fourier"] = ft

#%%

plt.plot(df['time_converted'], df['fourier'], color='green')
plt.xlabel('Time')
plt.ylabel('Fourier Transform')
plt.title('Fourier Transform of Precipitation')
plt.show()
#%%

ftt = np.fft.fft(df["tasmax"])
df["ftt"] = ftt
plt.plot(df['time_converted'], df['ftt'], color='red')
plt.xlabel('Time')
plt.ylabel('Fourier Transform')
plt.title('Fourier Transform of Temperature')
plt.show()

#%%
#%%
power_spectrum = np.abs(ft)**2
plt.plot(power_spectrum, color='green')
plt.xlabel('Frequency')
plt.ylabel('Power Spectrum')
plt.title('Power Spectrum of Precipitation')
plt.show()

#%%
power_spectrumt = np.abs(ftt)**2
plt.plot(power_spectrumt, color='red')
plt.xlabel('Frequency')
plt.ylabel('Power Spectrum')
plt.title('Power Spectrum of Temperature')
plt.show()
#%%
# using the scipy.signal.butter function to create a filter to get rid of the seasonal changes
# the filter is a low pass filter
# the cutoff frequency is 0.1
# the filter is a butterworth filter
# the order of the filter is 2

b, a = sig.butter(2, 0.1, btype='lowpass', analog=False, output='ba')

#%%
# using the scipy.signal.filtfilt function to apply the filter to the data
# the filter is applied twice to get rid of phase shift

pr_filt = sig.filtfilt(b, a, df["pr"])
tasmax_filt = sig.filtfilt(b, a, df["tasmax"])

#%%
# plotting the filtered data

plt.plot(df['time_converted'], pr_filt, color='green')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Precipitation Against Time')
plt.show()

#%%
plt.plot(df['time_converted'], tasmax_filt, color='red')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Against Time')
plt.show()

#%%
# plotting the underlying data and the filtered data on the same plot

plt.plot(df['time_converted'], df['pr'], color='green')
plt.plot(df['time_converted'], pr_filt, color='red')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Precipitation Against Time')
plt.show()

#%%
plt.plot(df['time_converted'], df['tasmax'], color='green')
plt.plot(df['time_converted'], tasmax_filt, color='red')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Against Time')
plt.show()

# %%
# removing linear trend from the data

pr_detrend = sig.detrend(df["pr"], type='linear')
tasmax_detrend = sig.detrend(df["tasmax"], type='linear')

#%%
# plotting the detrended data

plt.plot(df['time_converted'], pr_detrend, color='green')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Precipitation Against Time')
plt.show()

#%%
plt.plot(df['time_converted'], tasmax_detrend, color='red')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Against Time')
plt.show()

# %%
# plotting the underlying data and the detrended data on the same plot

plt.plot(df['time_converted'], df['pr'], color='green')
plt.plot(df['time_converted'], pr_detrend, color='red')
plt.xlabel('Time')
plt.ylabel('Precipitation')
plt.title('Precipitation Against Time')
plt.show()

# %%

plt.plot(df['time_converted'], df['tasmax'], color='green')
plt.plot(df['time_converted'], tasmax_detrend, color='red')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Temperature Against Time')
plt.show()
# %%
