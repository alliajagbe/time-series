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

plt.plot(df['time_converted'], df['fourier'])
plt.show()
#%%

ftt = np.fft.fft(df["tasmax"])
df["ftt"] = ftt
plt.plot(df['time_converted'], df['ftt'])
plt.show()




























