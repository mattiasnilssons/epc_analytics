#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 23:50:27 2021

@author: Mattias Nilsson
"""

import matplotlib                                      
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU 
import matplotlib.dates as mdates
import matplotlib.ticker as ticker                      
import seaborn as sns
import datetime as dt
import pandas as pd
import numpy as np

def do_plot(df_plot):
    # Plot context, this can stay in the block:
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('xtick', labelsize=14.5)
    plt.rc('ytick', labelsize=14) 
    plt.rc('axes', axisbelow=True, labelsize=14)
    plt.rc('legend', fontsize=10)
    myPalette = sns.color_palette("Spectral", as_cmap=True)
    sns.lineplot(x='timestamp', y='energy', data=df_plot, hue='meter_number', marker="o", legend='full', palette=myPalette)
    formatter = mdates.DateFormatter("%b %d") # date formats: https://strftime.org/

    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator()) 

    plt.ylabel('energy consumed, kWh', fontstyle='normal', fontweight='bold')
    plt.xlabel('time period', fontstyle='normal', fontweight='bold')
    plt.grid(True, which='both', color='whitesmoke')
    plt.legend(bbox_to_anchor=(1, 1), loc=2) 
    return df_plot

def classic_graph(df_day, 
                  yaxis_adjustment: float = 1):
    """
    

    Parameters
    ----------
    df_day : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.set_ylim(-0.0,21*yaxis_adjustment)
    ax2.set_ylim(-0.0,10.5*yaxis_adjustment)

    df_day.reset_index(inplace=True)
    
    df_day.timestamp = pd.to_datetime(df_day.timestamp)
    df_day.timestamp = np.int64(df_day.timestamp)
    df_day.timestamp = pd.to_datetime(df_day.timestamp)
    
    ax1.set_xlim([df_day.timestamp.dt.date.min(), df_day.timestamp.dt.date.max()])
    df_day.set_index('timestamp',inplace=True)
    ax1.grid(True, axis='y', color='gray')
    ax2.grid(False)
    
    ax1.bar(df_day.index, df_day.event_count, color='#F4BA52', label='events', zorder=2.52, linewidth=0, edgecolor='none', align='center',width=0.7)
    ax2.plot(df_day.index, df_day.meter_number, color='black', linewidth=2.58, label='no. of active meters', zorder=2.53)
    
    ax2.set_ylabel('no. of active households', fontstyle='normal', fontweight='bold', rotation=-90, labelpad=30)
    ax1.set_ylabel('no. of cooking events', fontstyle='normal', fontweight='bold', labelpad=15)
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')
    
    formatter = mdates.DateFormatter("%b %d") # date formats: https://strftime.org/
    ax1.xaxis.set_major_formatter(formatter)
    #ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(2*yaxis_adjustment))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1*yaxis_adjustment))
    ax1.tick_params(axis='x', bottom=True, zorder=1, rotation=0)
    ax1.tick_params(axis='y',length=0)
    ax1.tick_params(which='minor', length=0)
    ax2.tick_params(axis='y',length=0)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1*yaxis_adjustment))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.5*yaxis_adjustment))
    
    # Ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(lines + lines2, labels + labels2, handlelength=0.5, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    plt.savefig('images/day_plot.png', dpi=200)
    return plt.show()


def energy_and_time_plot(df_day, yaxis_adjustment: float = 1):
    """
    

    Parameters
    ----------
    df_day : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx() 
    
    df_day.reset_index(inplace=True)
    
    df_day.timestamp = pd.to_datetime(df_day.timestamp)
    df_day.timestamp = np.int64(df_day.timestamp)
    df_day.timestamp = pd.to_datetime(df_day.timestamp)
    
    ax1.set_xlim([df_day.timestamp.dt.date.min(), df_day.timestamp.dt.date.max()])
    
    ax2.bar(df_day.timestamp.dt.date.min(), 24*yaxis_adjustment, color='#F4BA52', alpha = 1.0, label=None, width = 0.4*yaxis_adjustment, zorder=11)
    ax2.bar(df_day.timestamp.dt.date.max(), 24*yaxis_adjustment, color='#DF1995', alpha = 1.0, label=None, width = 0.4*yaxis_adjustment, zorder=10)
    
    df_day.set_index('timestamp',inplace=True)
    
    ax1.set_ylim(0.001,24*yaxis_adjustment)
    ax2.set_ylim(0.001,8*yaxis_adjustment)
    
    ax1.grid(axis='y', which='both', color='lightgray')
    ax2.grid(False)
   
    ax1.bar(df_day.index, df_day.event_count, alpha = 1, color='#F4BA52', linewidth=0, edgecolor='black', label='events', zorder=7, align = 'edge', width=-0.2*yaxis_adjustment)
    ax2.bar(df_day.index, df_day.event_energy, alpha=1, color='#DF1995', label='energy', zorder=8, linewidth=0, edgecolor='black', width=0.2*yaxis_adjustment, align='edge')
    
    ax1.set_ylabel('no. of cooking events', fontstyle='normal', fontweight='bold', color = 'black', labelpad=15)
    ax2.set_ylabel('energy consumption, kWh', fontstyle='normal', fontweight='bold', rotation=-90, labelpad=35, color = 'black')
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')
    
    formatter = mdates.DateFormatter("%b %d") # date formats: https://strftime.org/
    ax1.xaxis.set_major_formatter(formatter)
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(6*yaxis_adjustment))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(3*yaxis_adjustment))
    ax1.tick_params(axis='x', bottom=True, rotation=0)
    ax1.tick_params(axis='y',length=0) 
    ax1.tick_params(which='minor', length=0, zorder=4)
    ax2.tick_params(axis='y',length=0, zorder=2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2*yaxis_adjustment))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1*yaxis_adjustment))
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(lines + lines2, labels + labels2, handlelength=0.7, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False) # Position of legend: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    plt.savefig('images/month_events.png', dpi=200)
    return plt.show()

def month_plot(df_month, norm_month_time, yaxis_adjustment: float=1):
    """
    

    Parameters
    ----------
    df_month : TYPE
        DESCRIPTION.
    norm_month_time : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    months = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    df_month['months'] = df_month.index.map(months)
    df_month['norm_month_time'] = df_month.index.map(norm_month_time)
    df_month['norm_event_count'] = df_month['event_count']/df_month['norm_month_time']
    df_month['norm_energy'] = df_month['event_energy']/df_month['norm_month_time']
    
    fig, ax1 = plt.subplots()
    
    ax2 = ax1.twinx()
    
    ax1.set_ylim(-0.0,500*yaxis_adjustment)
    ax2.set_ylim(-0.0,250*yaxis_adjustment)
    
    ax1.grid(True, axis='y', color='gray', zorder=1)
    ax2.grid(False)
    
    ax1.bar(df_month['months'], df_month['norm_event_count'], color='#F4BA52', label='events', zorder=8.52, linewidth=0, edgecolor='none', align='edge',width=-0.4)
    
    ax2.bar(df_month['months'], df_month['norm_energy'], color='#DE1995', label='energy', zorder=8.53, linewidth=0, edgecolor='none', align='edge',width=0.4)
    
    ax2.set_ylabel('aggregate energy consumption, kWh', fontstyle='normal', fontweight='bold', rotation=-90, labelpad=30)
    ax1.set_ylabel('no. of cooking events', fontstyle='normal', fontweight='bold', labelpad=15)
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')
    
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1)) # date tickers: https://matplotlib.org/3.1.1/api/dates_api.html
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(100*yaxis_adjustment))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(50*yaxis_adjustment))
    ax1.tick_params(axis='x', bottom=True, zorder=1, rotation=0)
    ax1.tick_params(axis='y',length=3)
    ax1.tick_params(which='minor', length=2)
    ax2.tick_params(axis='y',length=3)
    ax2.tick_params(which='minor', length=2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(50*yaxis_adjustment))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(25*yaxis_adjustment))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(lines + lines2, labels + labels2, handlelength=0.5, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    
    plt.title('Normalized cooking per month')
    plt.savefig('images/normalized_monthly_event_and_consumption.png', dpi=200)
    return plt.show()

def quantiles_plot(df_only_events, 
                   event_basis = None, 
                   yaxis_adjustment:float=1):
    """
    

    Parameters
    ----------
    df_only_events : TYPE
        DESCRIPTION.
    event_basis : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    df_only_events['ntimestamp'] = df_only_events['timestamp']
    df_only_events_meters = df_only_events.groupby(df_only_events.meter_number).agg({'energy':'max','event_energy':'sum', 'power_mean':'mean',\
                                                                        'event_count' : 'nunique', 'timestamp' : 'unique', 'ntimestamp' : 'nunique', 'cooking_time' :'unique'})
    df_only_events_meters.reset_index(inplace=True)
    df_only_events_meters.sort_values(by =['energy'], ascending=[True], inplace=True)
    df_only_events_meters['quantiles'] = pd.qcut(df_only_events_meters.energy, 4, labels=None)
    
    #df_only_events_meters.sort_values(by='energy',ascending=True,inplace=True)
    df_only_events_meters.reset_index(inplace=True)
    df_only_events_meters.reset_index(inplace=True)
    df_only_events_meters['level_0'] = df_only_events_meters['level_0'] + 1
    
    fig, ax1 = plt.subplots()
    
    if event_basis == None or event_basis == 'no' : 
        average_cooking_event = 1
        ax1.set_ylabel('energy per smart meter, kWh', fontstyle='normal', fontweight='bold', labelpad = 30)
    else: 
        average_cooking_event = (df_only_events_meters.event_energy.sum()/df_only_events_meters.event_count.sum())
        ax1.set_ylabel('no. of cooking events', fontstyle='normal', fontweight='bold', labelpad = 30)
    
    
    ax1.set_ylim(0.0,60/average_cooking_event*yaxis_adjustment)
    ax1.set_xlim([0, df_only_events_meters.meter_number.nunique()+1])
    
    ax1.grid(axis='y', which='major', color='lightgray')
    
    ax1.bar(df_only_events_meters.level_0, df_only_events_meters.energy/average_cooking_event, color='gray', alpha = 0.5, label=None, zorder=2.52, width = 0.8) # #F4BA52 gelb, #DF1995 a2ei pink
    
    ax1.bar(df_only_events_meters.level_0.quantile(.25), 450*yaxis_adjustment, color='yellow', alpha = 0.1, label='1st quartile, 25%', zorder=2.53, width = 0.8, linewidth=1, edgecolor='black')
    ax1.bar(df_only_events_meters.level_0.quantile(.50), 450*yaxis_adjustment, color='#F4BA52', alpha = 0.1, label='2nd quartile, 50%', zorder=2.53, width = 0.8, linewidth=1, edgecolor='black')
    ax1.bar(df_only_events_meters.level_0.quantile(.75), 450*yaxis_adjustment, color='darkorange', alpha = 0.1, label='3rd quartile, 75%', zorder=2.53, width = 0.8, linewidth=1, edgecolor='black')
    
    
    ax1.set_xlabel('Ranking of smart meters', labelpad=6, fontstyle='normal', fontweight='bold')
    
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(round(20/average_cooking_event)))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(round(10/average_cooking_event)))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    ax1.tick_params(axis='y',length=0)
    ax1.tick_params(which='major', length=2)
    ax1.tick_params(which='minor', length=0)
    ax1.tick_params(axis='x',length=2)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(lines, labels, handlelength=0.39, loc='upper center', bbox_to_anchor=(0.55, 1.15), ncol=4, frameon=False) # Position of legend: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    plt.savefig('images/quantiles_ongridpilot.png', dpi=200)
    return plt.show()




