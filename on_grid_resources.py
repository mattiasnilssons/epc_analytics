#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:53:35 2021

@author: Mattias Nilsson
"""
# Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU 
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, 
                               FormatStrFormatter, 
                               AutoMinorLocator) 
import matplotlib.ticker as ticker 

def combine_list_sm_data(df_epc, df_info):
    """

    Parameters
    ----------
    df_epc : Input is the data from the on-grid pilot.
    df_info : The lookup table for the on-grid pilot about the name and device type for each hh.

    Returns
    -------
    df_epc : Input is a Pandas DataFrame.

    """

    # Combine Lookup list with SM Data
    df_epc.reset_index(inplace=True)
    df_epc['device'] = df_epc.meter_number.map(
        df_info.set_index('meter_number')['device'].to_dict())
    df_epc['name'] = df_epc.meter_number.map(
        df_info.set_index('meter_number')['name'].to_dict())
    df_epc.set_index('timestamp', inplace=True)

    # Create a column 'sm_energy_total' to show how much energy that each
    # meter has consumed in total.
    sm_energy_total = df_epc.groupby('meter_number').agg(
        {'energy': 'max'}) - df_epc.groupby('meter_number').agg({'energy': 'min'})
    sm_energy_total.reset_index(inplace=True)
    df_epc['sm_energy_total'] = df_epc.meter_number.map(
        sm_energy_total.set_index('meter_number')['energy'].to_dict())

    return df_epc


def pie_chart(df_epc_energy, sizes, title, save_path):
    """

    Parameters
    ----------
    df_epc_energy : Input is a Pandas DataFrame.
    sizes : the size of each piece of the pie [read:category].
    title : str that is the title of the pie chart.
    save_path: name of directory and filename of picture.

    Returns
    -------
    Show the pie chart.

    """

    # Pie chart, where the slices will be ordered and plotted
    # counter-clockwise:
    labels = 'EPC', 'Hot plate', 'Kettle', 'Rice cooker'
    explode = (0.1, 0, 0, 0)  # only "explode" the 1st slice (i.e. 'EPC')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    plt.title(title)
    if save_path != None:
        plt.savefig(save_path, dpi=200)
    return plt.show()

def plot_on_grid_pilot(df_epc):
    df_plot = df_epc.copy()
    plt.rcParams['figure.figsize'] = [15, 5]
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rc('xtick', labelsize=14.5)
    plt.rc('ytick', labelsize=14) 
    plt.rc('axes', axisbelow=True, labelsize=14)
    plt.rc('legend', fontsize=10)
    df_plot.sort_values(by='energy_rank', inplace=True)
    myPalette = sns.color_palette("Spectral", as_cmap=True)
    sns.lineplot(x='timestamp', y='energy', data=df_plot, hue='legend_description', marker="o", legend='full', palette=myPalette)
    formatter = mdates.DateFormatter("%b %d") # date formats: https://strftime.org/
    
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    
    plt.ylabel('energy consumed, kWh', fontstyle='normal', fontweight='bold')
    plt.xlabel('time period', fontstyle='normal', fontweight='bold')
    plt.grid(True, which='both', color='whitesmoke')
    plt.legend(bbox_to_anchor=(1, 1), loc=2) 
    return plt.show()

def day_plot(df_only_events,
             yaxis_adjustment: float = 1):
    """
    Day plot

    Parameters
    ----------
    df_only_events : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ax1.set_ylim(-0.0001,100*yaxis_adjustment)
    ax2.set_ylim(0.0001,25*yaxis_adjustment)
    ax1.set_xlim([0, 24])
    #ax1.set_xlim([dt.date(1970, 1, 1), dt.date(1970, 2, 2)])
    
    ax1.grid(True, axis='y', color='gray')
    ax2.grid(False)
    
    ax1.bar(0, 100*yaxis_adjustment, color='#F4BA52', alpha = 1.0, label=None, width = 0.27)
    ax1.bar(24, 100*yaxis_adjustment, color='#DF1995', alpha = 1.0, label=None, width = 0.23, zorder=8)
    
    x1 = df_only_events.groupby([(df_only_events.timestamp.dt.hour)]).agg({'event_count':'nunique'}).index 
    y1 = df_only_events.groupby([(df_only_events.timestamp.dt.hour)]).agg({'event_count':'nunique'}).event_count
    
    ax1.bar(x1, y1, color='#F4BA52', label='events', zorder=2.52, linewidth=0, edgecolor='none', align='edge',width=-0.4)
    
    x2 = df_only_events.groupby([(df_only_events.timestamp.dt.hour)]).agg({'event_energy':'sum'}).index
    y2 = df_only_events.groupby([(df_only_events.timestamp.dt.hour)]).agg({'event_energy':'sum'}).event_energy
    
    #ax2.bar(df_hour_statistics_plot.index, df_hour_statistics_plot['change_meter_count'], color='none', alpha = 1, label='phantom energy', zorder=2.52, align='edge', linewidth=0.5, edgecolor='#DE1995',width=0.365)
    ax2.bar(x2, y2, color='#DE1995', label='energy', zorder=2.53, linewidth=0, edgecolor='none', align='edge',width=0.4)
    
    #ax2.plot(df_hour_statistics_plot.index, df_hour_statistics_plot['meter_number'], color='black', linewidth=2.58, label='households', zorder=2.54)
    ax2.set_ylabel('energy consumption, kWh', fontstyle='normal', fontweight='bold', rotation=-90, labelpad=30)
    ax1.set_ylabel('no. of cooking events', fontstyle='normal', fontweight='bold', labelpad=15)
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')
    
    ax1.xaxis.set_major_formatter(FormatStrFormatter('% 1.0f:00'))
    
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10*yaxis_adjustment))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(5*yaxis_adjustment))
    ax1.tick_params(axis='x', bottom=True, zorder=1, rotation=0)
    ax1.tick_params(axis='y',length=2)
    ax1.tick_params(which='minor', length=1)
    ax2.tick_params(axis='y',length=2)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2.5*yaxis_adjustment))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1.25*yaxis_adjustment))
    ax2.tick_params(which='minor', length=1)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    plt.legend(lines + lines2, labels + labels2, handlelength=0.5, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    
    plt.savefig('visuals_results/cooking24h.png', dpi=200)
    return plt.show()
 
def distr_plot(df_only_events):
    """
    

    Parameters
    ----------
    df_only_events : a Pandas Dataframe only containing grouped cooking events

    Yields
    ------
    Shows a distribution graph

    """
    fig, ax1 = plt.subplots()

    ax1.set_xlim(0.00,1.2)
    ax1.grid(axis='y', which='both', color='lightgray', zorder=3)

    # Create bar intervals
    def range_inc(start=0, stop=1.22, step=0.02):
        i = start
        while i <= stop:
            yield i
            i += step
    cuts = list(range_inc())
    cuts.extend([2, 10])

    df_overview = df_only_events.groupby(pd.cut(df_only_events['event_energy'],\
                                                             cuts)).agg({'event_count':'nunique','cooking_time':'mean'})

    del cuts[0]       
    df_overview['xaxis'] = cuts  

    ax1.bar(df_overview.xaxis, df_overview.event_count/df_overview.event_count.sum()*100, alpha = 1, color='#F4BA52', hatch="//", edgecolor='black', linewidth=1, label='energy consumption per cooking event, kWh', zorder=8, align = 'edge', width=0.02)
    ax1.set_ylabel('distribution', fontstyle='normal', fontweight='bold', color = 'black', labelpad=15)
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')

    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = ticker.FormatStrFormatter(fmt)
    ax1.yaxis.set_major_formatter(yticks)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1)) # date tickers: https://matplotlib.org/3.1.1/api/dates_api.html
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax1.tick_params(axis='x', bottom=True, rotation=0)
    ax1.tick_params(axis='y',length=0) 
    ax1.tick_params(which='minor', length=0, zorder=4)
    lines, labels = ax1.get_legend_handles_labels()

    plt.legend(lines, labels, handlelength=0.7, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False) # Position of legend: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot

    plt.savefig('visuals_results/distribution_energy_bars.png')
    return plt.show()

def distr_plot_time(df_only_events):
    fig, ax1 = plt.subplots()

    ax1.set_xlim(0,145)
    
    ax1.grid(axis='y', which='both', color='lightgray', zorder = 1)
    
    # Create bar intervals
    def range_inc(start=0, stop=120, step=2):
        i = start
        while i <= stop:
            yield i
            i += step
    cuts = list(range_inc())
    cuts.extend([140, 160,180,200,400])
    
    df_overview = df_only_events.groupby(pd.cut(df_only_events['cooking_time'],\
                                                             cuts)).agg({'event_count':'nunique','event_energy':'mean'})
    
    del cuts[0]       
    df_overview['xaxis'] = cuts          
    
    ax1.bar(df_overview.xaxis, df_overview.event_count/df_overview.event_count.sum()*100, alpha = 1, color='#DF1995', edgecolor='black', linewidth=1, label='cooking time, minutes', zorder=7, align = 'edge', width=2)
    ax1.set_ylabel('distribution', fontstyle='normal', fontweight='bold', color = 'black', labelpad=15)
    ax1.set_xlabel(None, labelpad=6, fontstyle='normal', fontweight='bold')
    
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = ticker.FormatStrFormatter(fmt)
    ax1.yaxis.set_major_formatter(yticks)
    
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10)) # date tickers: https://matplotlib.org/3.1.1/api/dates_api.html
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator()) 
    ax1.tick_params(axis='x', bottom=True, rotation=0)
    ax1.tick_params(axis='y',length=2) 
    ax1.tick_params(which='minor', length=2, zorder=4)
    lines, labels = ax1.get_legend_handles_labels()
    
    plt.legend(lines, labels, handlelength=0.7, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False) # Position of legend: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    plt.savefig('visuals_results/distr_cooking_time.png')
    return plt.show()

# Tables

def device_stats(df_epc, df_only_events):
    """

    Parameters
    ----------
    df_epc : TYPE
        DESCRIPTION.
    df_only_events : TYPE
        DESCRIPTION.

    Returns
    -------
    df_device_stats : TYPE
        DESCRIPTION.

    """
    f = {
            'energy_rank' : 'mean', 
            'cooking_time' : 'sum',
            'meter_number': 'mean',
            'current':'max'
    }

    g = df_epc.groupby(['device', 'event_count'])
    v1 = g.agg(f)
    v2 = g.agg(lambda x: x.drop_duplicates('event_energy', keep='first').event_energy.sum())

    df_device_stats = pd.concat([v1, v2.to_frame('event_energy')], 1)
    df_device_stats.reset_index(level=[0,1], inplace=True)
    df_device_stats = df_device_stats.groupby('device').agg({'event_count':'nunique','meter_number':'nunique','event_energy':'sum','current':'max'})
    df_device_stats['events/meter'] = df_device_stats.event_count/df_device_stats.meter_number
    df_device_stats['energy/meter'] = df_device_stats.event_energy/df_device_stats.meter_number
    df_device_stats.rename(
        columns={'event_energy': 'total_event_energy',
                 'current': 'max_current',
            'meter_number': 'no_of_meters'}, inplace=True)
    df_device = df_only_events.groupby('device').agg({'event_energy':'mean', 'cooking_time':'mean','event_count':'nunique'})
    df_device.rename(
        columns={'event_energy': 'average_event_energy',
            'cooking_time': 'average_cooking_time'}, inplace=True)
    df_device.drop('event_count',axis=1,inplace=True)
    df_device_stats = pd.concat([df_device_stats,df_device],1)
    return df_device_stats
