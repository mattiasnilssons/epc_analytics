#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 08:30:08 2020

@author: Mattias Nilsson
"""
# Packages
import pandas as pd
import numpy as np


# 9 different functions
# Naming of the functions are not clear 
# (2 are resolving some issue) One function for each purpose.
# (4 functions are fixing cooking event) 
# (2-3 functions that are making output/dataframes) Naming and grouping
# Start & End should be in the same row
# Longer lines of code than possible
# I would like to have Unit Tests.
# Commits different names, not misleading, clear and precise. The longer the name for longer functions, because the parameters are usually more important.
# Standard for naming the commits with some detail. What did you do this and how did you do this. Added upper limit to cooking time for an event.
# Shorter extract_cooking_events function by a max


def resolve_spikes(df_raw,
                   minimum_size_of_spikes: float = 1,
                   time_resolution=None,
                   minimum_energy_per_cooking_event=None,
                   minimum_power_mean=None,
                   minimum_active_load=None,
                   power_capacity=None,
                   t_between=None,
                   minimum_event_current=None,
                   maximum_cooking_time=None,
                   energy_error_margin=None):
    """
    Check a Pandas Dataframe called 'df_raw' for data spikes larger than 1 kWh
    and handles them accordingly to return the updated dataframe.

    Parameters
    ----------
    df_raw : a dataframe that needs to be sorted by meter_number and timestamp.
    mininum_size_of_spikes : float, min. energy consumption of a data spike.
    Explained below in points 1-3.. The default is 1.

    (1) Why range(100)?: There are more than 50 recordings in some of the spikes.
    (2) Why not drop rows?: To save the dates of when the spikes occur.
    (3) Why mininum_size_of_spikes? = To avoid losing small energy movements, incl. timestamp issue.

    Returns
    -------
    df_raw : returns a Pandas DataFrame.

    """

    for _ in range(100):
        df_raw.loc[(df_raw.energy > df_raw.energy.shift(-1) + minimum_size_of_spikes) &
                   (df_raw.meter_number == df_raw.meter_number.shift(-1)), 'energy'] = df_raw.energy.shift(-1)
    return df_raw


def extract_cooking_events(df_raw,
                           minimum_energy_per_cooking_event: float = 0.05,
                           minimum_power_mean: float = 0.05,
                           minimum_event_current: float = 2,
                           maximum_cooking_time: float = 300,
                           time_resolution: int = 5,
                           mininum_size_of_spikes=None,
                           minimum_active_load: float = 0.15,
                           power_capacity: float = 1,
                           t_between: float = 15,
                           energy_error_margin=None):
    """
    Extract cooking events by defining start and end of each cooking event,
    disqualify cooking events that aren't using enough energy and returns
    a Pandas DataFrame.

    Parameters
    ----------
    df_raw : a Pandas Dataframe that needs to be sorted by meter_number and timestamp.
    minimum_energy_per_cooking_event : float, min. energy consumption of a cooking event,
    excl. start & end energy (recorded within the time resolution interval) that will be added later.
    The default is 0.05.
    minimum_power_mean : float, disqualify cooking events that have an average power
    level below e.g. 50 W which is below MCR 5% of an EPC. The default is 0.05.
    minimum_event_current : float, disqualify cooking events that don't have a recorded current above
    a certain threshold, e.g. above 1 A. An EPC normally uses approx. 4 A. The default is 0.5.
    maximum_cooking_time : float, the maximum length [minutes] that an EPC is believed to be cooking before 
    it's not considered a normal EPC event. The default is 300. 
    time_resolution : int, for DESCRIPTION see function 'apply_event_conditions'. The default is 5.
    minimum_active_load : float, for DESCRIPTION see function 'apply_event_conditions'. The default is 0.15.
    power_capacity : float, for DESCRIPTION see function 'apply_event_conditions'. The default is 1.
    t_between : float, for DESCRIPTION see function 'apply_event_conditions'. The default is 15.

    Returns
    -------
    df_processed : returns a Pandas DataFrame.

    """

    df_processed = df_raw.copy()

    # Recalculate the power level of the EPC using the current and voltage,
    # because this is originally calculated using an average value which can
    # have a delay.
    df_processed['power'] = (df_processed.current *
                             df_processed.voltage) / 1000  # kW

    # Format the 'timestamp' column
    format_timestamp(df_processed)
    df_processed.reset_index(inplace=True)

    # Create columns based on columns 'meter_number' and 'timestamp' by
    # selecting the time difference between rows for each meter_number to
    # conduct the further analysis.
    df_processed.loc[(df_processed.meter_number.diff() == 0),
                     'diff_prev_timestamp'] = df_processed.timestamp.diff()
    df_processed.loc[(df_processed.meter_number.diff(-1) == 0),
                     'diff_next_timestamp'] = df_processed.timestamp.shift(-1) - df_processed.timestamp

    # Create columns for Cooking 'start' & 'end'
    df_processed['cooking_start'] = False
    df_processed['cooking_end'] = False

    # Apply conditions for creating distinct cooking events
    df_processed = apply_event_conditions(df_processed,
                                          time_resolution,
                                          minimum_active_load,
                                          power_capacity,
                                          t_between)

    # Create a column called 'cooking_event' for accumulated numbering of
    # cooking events.
    df_processed['cooking_event'] = 0
    df_processed.cooking_event += df_processed['cooking_start']
    df_processed.cooking_event = df_processed['cooking_event'].cumsum()

    # Create columns to show start timestamp of each cooking event.
    start_cooking = df_processed.groupby('cooking_event').first()
    start_cooking.reset_index(inplace=True)
    df_processed['time_start'] = df_processed.cooking_event.map(
        start_cooking.set_index('cooking_event')['timestamp'].to_dict())
    df_processed['energy_start'] = df_processed.cooking_event.map(
        start_cooking.set_index('cooking_event')['energy'].to_dict())

    # Create columns to show end timestamp of each cooking event.
    end_cooking = df_processed.copy()
    end_cooking = df_processed.groupby('cooking_event').last()
    end_cooking.reset_index(inplace=True)
    df_processed['time_end'] = df_processed.cooking_event.map(
        end_cooking.set_index('cooking_event')['timestamp'].to_dict())
    df_processed['energy_end'] = df_processed.cooking_event.map(
        end_cooking.set_index('cooking_event')['energy'].to_dict())

    # Create columns for getting duration of cooking event and sequence time
    # during cooking event.
    df_processed['cooking_time'] = (
        df_processed.time_end - df_processed.time_start) / np.timedelta64(1, 'm')
    df_processed['seq_time'] = (
        df_processed.timestamp - df_processed.time_start) / np.timedelta64(1, 'm')

    # Disqualify cooking events of 'too low' average energy
    df_processed.loc[((df_processed.energy_end -
                       df_processed.energy_start < minimum_energy_per_cooking_event) | ((df_processed.energy_end -
                                                                          df_processed.energy_start) /
                                                                         (df_processed.cooking_time /
                                                                          60) < minimum_power_mean)), 'cooking_event'] = np.nan

    # Classify recordings that are not part of a cooking event with NaN.
    df_processed.loc[((df_processed.cooking_event.isnull())
                      ), 'cooking_time'] = np.nan
    df_processed.loc[((df_processed.cooking_event.isnull())
                      ), 'seq_time'] = np.nan

    # Check the max current, which is approx. 4 A for the EPC.
    # If it's much different, this can prove that other device is used.
    df_processed_current = df_processed.groupby(
        'cooking_event').agg({'current': 'max'})
    df_processed_current.reset_index(inplace=True)
    df_processed['event_max_current'] = df_processed.cooking_event.map(
        df_processed_current.set_index('cooking_event')['current'].to_dict())

    # Disqualify cooking events that doesn't meet criterium for current
    # threshold
    df_processed.loc[
        (
            ((df_processed.event_max_current < minimum_event_current)
             & (df_processed.event_max_current != 0))
            | (df_processed.cooking_time > maximum_cooking_time)
            | (df_processed.energy_end - df_processed.energy_start < minimum_energy_per_cooking_event)
            
        ), 'cooking_event'] = np.nan

    # Set timestamp in index to facilitate plotting with timeseries on the
    # x-axis.
    df_processed.set_index('timestamp', inplace=True)

    return df_processed


def apply_event_conditions(df_processed,
                           time_resolution: int = 5,
                           minimum_active_load: float = 0.15,
                           power_capacity: float = 1,
                           t_between: float = 15,
                           mininum_size_of_spikes=None,
                           minimum_energy_per_cooking_event=None,
                           minimum_power_mean=None,
                           minimum_event_current=None,
                           maximum_cooking_time=None,
                           energy_error_margin=None):
    """
    Apply event conditions to get True or False values in the 'cooking_start'
    and 'cooking_false' columns and returns a Pandas DataFrame.

    Parameters
    ----------
    df_processed : a Pandas Dataframe that is used to extract cooking events.
    time_resolution : int, 1\5 minutes. Indicates the setting for the time resolution
    of the smart meter in minutes. The default is 5.
    minimum_active_load : float, min. power load as a percentage of the power capacity to
    indicate that an EPC is active, i.e. 0.15 = 15 % of Max. Rated Capacity. The default is 0.15.
    power_capacity : the power capacity of the EPC in this project, which is 1 kW. The default is 1.
    t_between : float, the max. time interval between two recordings which indicates
    that they belong to the same cooking event. The default is 15.

    Returns
    -------
    df_processed : returns a Pandas DataFrame with the indications of start and end of cooking events.

    """

    # (i): create coefficients to indicate when an EPC is turned ON
    power_threshold = minimum_active_load * power_capacity
    energy_threshold = power_threshold * time_resolution / 60

    # (ii): Create column 'load' for when a load is applied as the energy
    # consumption [kWh] between the current and pervious recording.
    df_processed.loc[(
        (
            (df_processed.energy.diff() > energy_threshold)
            | (df_processed.power > minimum_active_load * power_capacity))
        & (df_processed.meter_number == df_processed.meter_number.shift())
    ), 'load'] = df_processed.energy.diff()

    # (iii): Create a column 'load_count' for accumulated numbering of when a load is applied
    df_processed['load_count'] = 0  # start
    df_processed.loc[(df_processed.load.isnull()
                      == False), 'load_count'] += 1
    df_processed.load_count = df_processed.load_count.cumsum()

    # (iv): Create a column 'timestamp_load' for a timestamp of each load instance
    load_instance = df_processed.groupby('load_count').first()
    load_instance.reset_index(inplace=True)
    df_processed['timestamp_load'] = df_processed.load_count.map(
        load_instance.set_index('load_count')['timestamp'].to_dict())

    # (v): Cooking_start = TRUE: if timestamp_load - current timestamp is more
    # than t_between and above energy_threshold OR new meter_number.
    df_processed.loc[
        (
            (
                (df_processed.timestamp -
                 df_processed.timestamp_load.shift() > pd.to_timedelta(
                     t_between,
                     unit='m'))
                & (df_processed.energy.diff() >= energy_threshold))
            | (
                df_processed.meter_number != df_processed.meter_number.shift())
        ), 'cooking_start'] = True

    # (vi): Cooking_start = FALSE: if energy increase is above energy threshold and diff_prev_timestamp is less than t_between.
    df_processed.loc[
        (
            (df_processed.energy.diff() >= energy_threshold)
            & (df_processed.diff_prev_timestamp < pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_start'] = False

    # (vii): Cooking_start = TRUE: if previous to current timestamp_load difference is above t_between + time_resolution AND power level is above 'power threshold', i.e. minimum_active_load * power_capacity
    df_processed.loc[
        (
            (df_processed.timestamp_load.diff() > pd.to_timedelta(t_between + time_resolution, unit='m'))
            & (df_processed.power >= power_threshold)
        ), 'cooking_start'] = True

    # (viii): Cooking_start = FALSE: if a recording in the previous-to-previous row is within t_between from current row, make sure that this isn't start of a cooking event since it should be part of the same event.
    df_processed.loc[
        (
            (df_processed.timestamp.diff(2) <= pd.to_timedelta(t_between, unit='m'))
            & (df_processed.cooking_start.shift(2))
            & (df_processed.cooking_start)
            & (df_processed.meter_number == df_processed.meter_number.shift(2))
        ), 'cooking_start'] = False

    # (ix): Cooking_end = TRUE: if difference between current timestamp and timestamp_load is above t_between AND power is above threshold on current and previous row AND same meter_number are all TRUE.
    df_processed.loc[
        (
            (df_processed.timestamp - df_processed.timestamp_load > pd.to_timedelta(
                t_between, unit='m'))
            & ((df_processed.power < power_threshold)
               & (df_processed.power.shift() < power_threshold)
               & (df_processed.meter_number == df_processed.meter_number.shift())
               )
            | (df_processed.energy - df_processed.energy.shift(-1) == 0)
        ), 'cooking_end'] = True

    # (x): Cooking_end = TRUE: if cooking_start in next row is TRUE OR new meter_number
    df_processed.loc[
        (
            (df_processed.cooking_start.shift(-1))
            | (df_processed.meter_number != df_processed.meter_number.shift(-1))
        ), 'cooking_end'] = True

    # (xi): Cooking_end = TRUE: if cooking_end in next row is TRUE AND diff_next_timestamp is above t_between
    df_processed.loc[
        (
            (df_processed.cooking_end.shift(-1))
            & (df_processed.diff_next_timestamp > pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = True

    # (xii): Cooking_start = TRUE: if cooking_end on prev row AND cooking_end on current row
    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
        ), 'cooking_start'] = True

    # (ix): Cooking_start = TRUE: if difference between current timestamp and timestamp_load is above t_between AND power above power_threshold
    df_processed.loc[
        (
            (df_processed.timestamp - df_processed.timestamp_load > pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.power >= power_threshold)
        ), 'cooking_start'] = True

    # (xiv): Cooking_end = FALSE: if cooking_end on prev row AND cooking_end in current row == TRUE AND diff_prev_timestamp is more than t_between AND diff_next_timestamp is less than t_between.
    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
            & (df_processed.diff_prev_timestamp > pd.to_timedelta(t_between, unit='m'))
            & (df_processed.diff_next_timestamp <= pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = False

    # (xvi): Cooking_end = TRUE: if previous and next row isn't an end of a cooking event and the time to previous recording is more than t_between, then cooking_end = True.
    df_processed.loc[
        (
            (df_processed.cooking_end.shift(-1) == 0)
            & (df_processed.cooking_end.shift() == 0)
            & (df_processed.diff_next_timestamp > pd.to_timedelta(
                t_between, unit='m'))
        ), 'cooking_end'] = True

    # (xv): Cooking_start = FALSE: if cooking_start on prev row AND cooking_start = TRUE in current row AND diff_prev_timestamp is less than t_between AND diff_next_timestamp is more than t_between.
    df_processed.loc[
        (
            (df_processed.cooking_start.shift())
            & (df_processed.cooking_start)
            & (df_processed.diff_prev_timestamp <= pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.diff_next_timestamp > pd.to_timedelta(
                t_between, unit='m'))
        ), 'cooking_start'] = False

    # (xvi): Cooking_start = TRUE: if previous timestamp is more than t_between + time_resolution, energy consumption is above 0,
    # time to next timestamp is less than t_between and the meter number is
    # the same in the previous and next row, then cooking_start = TRUE.
    df_processed.loc[
        (
            (df_processed.diff_prev_timestamp > pd.to_timedelta(t_between + time_resolution, unit='m'))
            & (df_processed.energy.shift(-1) - df_processed.energy != 0)
            & (df_processed.diff_next_timestamp <= pd.to_timedelta(t_between, unit='m'))
            & (df_processed.meter_number == df_processed.meter_number.shift())
            & (df_processed.meter_number == df_processed.meter_number.shift(-1))
        ), 'cooking_start'] = True

    # (xvii): Cooking_start = FALSE: if cooking_start in prev row = TRUE AND cooking_start in current row = TRUE AND diff_prev_timestamp is less than t_between AND prev row has power above threshold.
    df_processed.loc[
        (
            (df_processed.cooking_start.shift())
            & (df_processed.cooking_start)
            & (df_processed.diff_prev_timestamp < pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.power.shift() >= power_threshold)
        ), 'cooking_start'] = False

    # (xviii): if new meter number Cooking_start = TRUE, Cooking_end = FALSE
    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_start'] = True
    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_end'] = False

    # (xix): Take out "unnecessary" cooking_end
    df_processed.loc[
        (
            (
                (df_processed.cooking_start.shift(-1) == 0)
                & (df_processed.cooking_start.shift() == 0)
                & (df_processed.cooking_start == 0)
            )
            | (df_processed.cooking_start == 1)

        ), 'cooking_end'] = False
    return df_processed


def resolve_timestamps(df_processed,
                       energy_error_margin: float = 0.04,
                       time_resolution=None,
                       minimum_active_load=None,
                       power_capacity=None,
                       t_between=None,
                       mininum_size_of_spikes=None,
                       minimum_energy_per_cooking_event=None,
                       minimum_power_mean=None,
                       minimum_event_current=None,
                       maximum_cooking_time=None):
    """
    Resolve timestamps is a function that is removing cooking events that have been duplicated.
    This occured because the smart meters needed to reconfigure the factory settings, which had the local timezone, to UTC time.
    The timestamp duplicates are identified by comparing the energy meter value [kWh] at the start and end of each cooking events per meter.
    If the difference between cooking events in the energy meter value [kWh] at the start or end is less than energy_error_margin, this indicates that
    this cooking event is already classified as a cooking event and that the recording is a duplicate.
    All smart meters are now in UTC by default to avoid this problem happening in the future, i.e. since Dec-2020.

    Parameters
    ----------
    df_processed : a Pandas Dataframe that has defined cooking events.
    energy_error_margin : float, the energy consumption difference [kWh] that the previous
    cooking event should have compared to current cooking event. The default is 0.04.

    Returns
    -------
    df_epc : returns a Pandas DataFrame without any duplicated cooking events.

    """

    # Make a copy of dataframe to differentiate between the Pandas DataFrame with
    # vis-a-vis without duplicated cooking events.
    df_epc = df_processed.copy()
    df_epc.reset_index(inplace=True)

    # Make an intermediate Cooking Event Count
    df_epc['event_count'] = 0
    df_epc.loc[(df_epc.cooking_event.diff()
                != 0), 'event_count'] += 1
    df_epc.event_count = df_epc['event_count'].cumsum()
    df_epc.loc[
        (df_epc.cooking_event.isnull()),
        'event_count'] = np.nan

    # Check start of events
    start_of_event = df_epc.copy()
    start_of_event = start_of_event.groupby(
        ['meter_number', 'event_count']).head(1)
    start_of_event.loc[
        ((start_of_event['energy'] -
          energy_error_margin <= start_of_event['energy'].shift()) & (
            start_of_event.event_count.isnull() == False) & (
            start_of_event.meter_number == start_of_event.meter_number.shift())),
        'timestamp_issue'] = True

    # Make an indication of timestamp issue at start of cooking event.
    df_epc['timestamp_issue'] = df_epc.event_count.map(
        start_of_event.set_index('event_count')['timestamp_issue'].to_dict())

    # Check end of events
    end_of_event = df_epc.copy()
    end_of_event = end_of_event.groupby(
        ['meter_number', 'event_count']).tail(1)
    end_of_event.loc[
        ((end_of_event['energy'] -
          energy_error_margin <= end_of_event['energy'].shift()) & (
            end_of_event.event_count.isnull() == False) & (
            end_of_event.meter_number == end_of_event.meter_number.shift())),
        'timestamp_issue'] = True

    # Make an indication of timestamp issue at end of cooking event.
    df_epc['timestamp_issue'] = df_epc.event_count.map(
        end_of_event.set_index('event_count')['timestamp_issue'].to_dict())

    # Discard recordings that are part of duplicated cooking events,
    # only leaving the cooking event's first occurance.
    df_epc.drop(df_epc[(df_epc['timestamp_issue'] == 1)].index, inplace=True)

    # Update the cooking event count
    df_epc['cooking_event'] = 0
    df_epc.loc[((df_epc.event_count.diff() != 0) & (
        df_epc.event_count.isnull() == False)), 'cooking_event'] += 1
    df_epc.cooking_event = df_epc['cooking_event'].cumsum()

    # Drop rows that do no longer have any function
    df_epc.drop(['event_count', 'timestamp_issue'], axis=1, inplace=True)

    # Set timestamp in index to facilitate plotting with timeseries on the
    # x-axis.
    df_epc.set_index('timestamp', inplace=True)
    return df_epc


def add_event_end(df_epc,
                  time_resolution: int = 5,
                  power_capacity: float = 1,
                  energy_error_margin=None,
                  minimum_active_load=None,
                  t_between=None,
                  mininum_size_of_spikes=None,
                  minimum_energy_per_cooking_event=None,
                  minimum_power_mean=None,
                  minimum_event_current=None,
                  maximum_cooking_time=None):
    """
    Add a row at the end of each cooking event that meets certain criteria.
    This new row will compensate for the energy consumed at the end of a cooking event, i.e. after the last recording of the "originally defined" cooking event.
    The added energy comes from the energy value difference [kWh] between the current row and the next row.
    This energy value difference occurs due to the EPC being turned off prematurely or because a data gap/loss-of-network occurs.

    Parameters
    ----------
    df_epc : a Pandas Dataframe that has defined cooking events.
    time_resolution : int, 1\5 minutes. Indicates the setting for the time resolution
    of the smart meter in minutes. The default is 5.
    power_capacity : float, the power capacity of the EPC in this project,
    which is 1 kW. The default is 1.

    Returns
    -------
    df_epc : returns a Pandas DataFrame which includes the prolonged start of
    certain cooking events.

    """

    # Make sure 'timestamp' column is not assigned as index
    df_epc = format_timestamp(df_epc)
    df_epc.reset_index(inplace=True)

    # Make a copy of the dataframe which will used to mark the end of each
    # cooking event
    df_epc_energy_gaps = df_epc.copy()

    # Quantify the energy gap at the end of each cooking event
    df_epc_energy_gaps.loc[
        (
            (df_epc_energy_gaps.cooking_start == False)
            & (df_epc_energy_gaps.cooking_end == True)
            & (
                (df_epc_energy_gaps.cooking_event != df_epc_energy_gaps.cooking_event.shift())
                | (df_epc_energy_gaps.cooking_event != df_epc_energy_gaps.cooking_event.shift(-1))
            )
            & (df_epc_energy_gaps.meter_number == df_epc_energy_gaps.meter_number.shift(-1))
        ), 'energy_gap_to_next'] = df_epc_energy_gaps.energy.shift(-1) - df_epc_energy_gaps.energy

    # Only keep the last row of each cooking event that has an energy_gap
    # which should be used to prolong the cooking event.
    df_epc_energy_gaps = df_epc_energy_gaps.groupby('cooking_event').last()
    df_epc_energy_gaps = df_epc_energy_gaps.loc[df_epc_energy_gaps['energy_gap_to_next'] > 0]

    # Reset the index to keep the 'cooking_event' in the columns
    df_epc_energy_gaps.reset_index(inplace=True)

    # Equalize the energy gap in kWh to the equivalent in minutes
    df_epc_energy_gaps['energy_gap_time'] = df_epc_energy_gaps.energy_gap_to_next / \
        power_capacity * 60

    # Format the value in minutes to allow timedelta calculations
    df_epc_energy_gaps['energy_gap_time_datetime'] = df_epc_energy_gaps['energy_gap_time'] * \
        60 * np.timedelta64(1, 's')

    # Check the occurences of energy gaps with a timedelta below time
    # resolution, e.g. 5 minutes
    energy_gap_less_than_time_resolution = (
        df_epc_energy_gaps.energy_gap_time <= time_resolution)

    # Check the occurences of energy gaps with a timedelta above time
    # resolution, e.g. 5 minutes
    energy_gap_above_time_resolution = (
        df_epc_energy_gaps.energy_gap_time > time_resolution)

    # Update row of extended cooking event at ending, comprising updating 6 columns
    # If the energy gap is less than the time resolution value.
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'timestamp'] += df_epc_energy_gaps.energy_gap_time_datetime
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'time_end'] += df_epc_energy_gaps.energy_gap_time_datetime
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'cooking_time'] += df_epc_energy_gaps.energy_gap_time
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'seq_time'] += df_epc_energy_gaps.energy_gap_time
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'energy'] += df_epc_energy_gaps.energy_gap_to_next
    df_epc_energy_gaps.loc[energy_gap_less_than_time_resolution,
                           'energy_end'] += df_epc_energy_gaps.energy_gap_to_next

    # If the energy gap is above than the time resolution value.
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'timestamp'] += pd.Timedelta(minutes=time_resolution)
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'time_end'] += pd.Timedelta(minutes=time_resolution)
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'cooking_time'] += time_resolution
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'seq_time'] += time_resolution
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'energy'] += time_resolution / 60
    df_epc_energy_gaps.loc[energy_gap_above_time_resolution,
                           'energy_end'] -= time_resolution / 60

    # Add the new rows to the main Pandas DataFrame
    df_epc = df_epc.append(df_epc_energy_gaps)

    # Remove "unnecessary" rows
    df_epc.drop(['energy_gap_time', 'energy_gap_time_datetime',
                 'energy_gap_to_next'], axis=1, inplace=True)

    # Update all rows of each cooking event that has been prolonged
    df_epc['time_end'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['time_end'].to_dict())
    df_epc['cooking_time'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['cooking_time'].to_dict())
    df_epc['energy_end'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['energy_end'].to_dict())
    df_epc.sort_values(by=['meter_number', 'timestamp'],
                       ascending=[True, True], inplace=True)

    # Set timestamp in index to facilitate plotting with timeseries on the
    # x-axis.
    df_epc.set_index('timestamp', inplace=True)
    return df_epc


def add_event_start(df_epc,
                    time_resolution: int = 5,
                    power_capacity: float = 1,
                    energy_error_margin=None,
                    minimum_active_load=None,
                    t_between=None,
                    mininum_size_of_spikes=None,
                    minimum_energy_per_cooking_event=None,
                    minimum_power_mean=None,
                    minimum_event_current=None,
                    maximum_cooking_time=None):
    """
    Add a row at the start of each cooking event that meets certain criteria.
    This new row will compensate for the energy consumed at the start of a cooking event that happens before the first "original" recording in a defined cooking event.

    Parameters
    ----------
    df_epc : a Pandas Dataframe that has defined cooking events.
    time_resolution : int, 1\5 minutes. Indicates the setting for the time resolution
    of the smart meter in minutes. The default is 5.
    power_capacity : float, the power capacity of the EPC in this project,
    which is 1 kW. The default is 1.

    Returns
    -------
    df_epc : returns a Pandas DataFrame which includes the prolonged start of
    certain cooking events.

    """

    # Make sure 'timestamp' column is not assigned as index
    df_epc = format_timestamp(df_epc)
    df_epc.reset_index(inplace=True)

    # Make a copy of the dataframe which will used to mark the start of each
    # cooking event
    df_epc_energy_gaps2 = df_epc.copy()

    # Quantify the energy gap at the start of each cooking event
    df_epc_energy_gaps2.loc[
        (
            (df_epc_energy_gaps2.cooking_end == False)
            & (df_epc_energy_gaps2.cooking_start == True)
            & (
                (df_epc_energy_gaps2.cooking_event != df_epc_energy_gaps2.cooking_event.shift())
                | (df_epc_energy_gaps2.cooking_event != df_epc_energy_gaps2.cooking_event.shift(-1))
            )
            & (df_epc_energy_gaps2.energy.shift().isnull())
            & (df_epc_energy_gaps2.meter_number == df_epc_energy_gaps2.meter_number.shift())
        ), 'energy_gap_to_prev'] = df_epc_energy_gaps2.energy.diff()

    # Only keep the first row of each cooking event that has an energy_gap
    # which should be used to prolong the cooking event.
    df_epc_energy_gaps2 = df_epc_energy_gaps2.groupby('cooking_event').first()
    df_epc_energy_gaps2 = df_epc_energy_gaps2.loc[df_epc_energy_gaps2['energy_gap_to_prev'] > 0]

    # Reset the index to keep the 'cooking_event' in the columns
    df_epc_energy_gaps2.reset_index(inplace=True)

    # Equalize the energy gap in kWh to the equivalent in minutes
    df_epc_energy_gaps2['energy_gap_time'] = df_epc_energy_gaps2.energy_gap_to_prev / \
        power_capacity * 60

    # Format the value in minutes to allow timedelta calculations
    df_epc_energy_gaps2['energy_gap_time_datetime'] = df_epc_energy_gaps2['energy_gap_time'] * \
        60 * np.timedelta64(1, 's')

    # Check the occurences of energy gaps with a timedelta below time
    # resolution, e.g. 5 minutes
    energy_gap_less_than_time_resolution = (
        df_epc_energy_gaps2.energy_gap_time <= time_resolution)

    # Check the occurences of energy gaps with a timedelta above time
    # resolution, e.g. 5 minutes
    energy_gap_above_time_resolution = (
        df_epc_energy_gaps2.energy_gap_time > time_resolution)

    # Update row of extended cooking event at beginning, comprising updating 6 columns
    # If the energy gap is less than the time resolution value.
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'timestamp'] -= df_epc_energy_gaps2.energy_gap_time_datetime
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'time_start'] -= df_epc_energy_gaps2.energy_gap_time_datetime
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'cooking_time'] += df_epc_energy_gaps2.energy_gap_time
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'seq_time'] -= df_epc_energy_gaps2.energy_gap_time
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'energy'] -= df_epc_energy_gaps2.energy_gap_to_prev
    df_epc_energy_gaps2.loc[energy_gap_less_than_time_resolution,
                            'energy_start'] -= df_epc_energy_gaps2.energy_gap_to_prev

    # If the energy gap is above than the time resolution value.
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'timestamp'] -= pd.Timedelta(minutes=time_resolution)
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'time_start'] -= pd.Timedelta(minutes=time_resolution)
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'cooking_time'] += time_resolution
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'seq_time'] -= time_resolution
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'energy'] -= time_resolution / 60
    df_epc_energy_gaps2.loc[energy_gap_above_time_resolution,
                            'energy_start'] -= time_resolution / 60

    # Add the new rows to the main Pandas DataFrame
    df_epc = df_epc.append(df_epc_energy_gaps2)

    # Remove "unnecessary" rows
    df_epc.drop(['energy_gap_time', 'energy_gap_time_datetime',
                 'energy_gap_to_prev'], axis=1, inplace=True)

    # Update all rows of each cooking event that has been prolonged
    df_epc['time_start'] = df_epc.cooking_event.map(
        df_epc_energy_gaps2.set_index('cooking_event')['time_start'].to_dict())
    df_epc['cooking_time'] = df_epc.cooking_event.map(
        df_epc_energy_gaps2.set_index('cooking_event')['cooking_time'].to_dict())
    df_epc['energy_start'] = df_epc.cooking_event.map(
        df_epc_energy_gaps2.set_index('cooking_event')['energy_start'].to_dict())
    df_epc.sort_values(by=['meter_number', 'timestamp'],
                       ascending=[True, True], inplace=True)

    # Set timestamp in index to facilitate plotting with timeseries on the
    # x-axis.
    df_epc.set_index('timestamp', inplace=True)
    return df_epc


def create_only_event_df(df_epc,
                         TZS_per_kWh: int = 100):
    """
    Create a Pandas DataFrame which only contains rows of cooking events by grouping
    all recordings that belongs to a qualified cooking event.


    Parameters
    ----------
    df_epc : a dataframe that has defined cooking events.
    TZS_per_kWh : int, the price of cooking with the EPC per kWh in Tanzanian Shilling.
    The default is 100.

    Returns
    -------
    df_only_events : returns a dataframe without any non-cooking events.

    """

    # Make a copy of dataframe to keep the non-cooking events in the dataframe
    # that is used as source file.
    df_only_events = df_epc.copy()
    df_only_events.reset_index(inplace=True)

    # Create a new column called 'event_energy' to do statistics of how much
    # energy consumption that each cooking event has.
    df_only_events['event_energy'] = df_only_events.energy

    # Group all cooking events according to columns 'meter_number' and
    # 'cooking_event'
    df_only_events = df_only_events.groupby(['meter_number',
                                             'cooking_event']).agg({'energy': 'max',
                                                                    'event_energy': 'min',
                                                                    'power': 'mean',
                                                                    'cooking_time': 'max',
                                                                    'timestamp': 'min',
                                                                    'current': 'max',
                                                                    'voltage': 'count',
                                                                    'id': 'mean'})

    # (a): Count the number of recordings in each cooking event
    df_only_events.rename(
        columns={
            'voltage': 'no_recordings'},
        inplace=True)

    # (b): Calculate the energy usage of each cooking event
    df_only_events.event_energy = df_only_events.energy - df_only_events.event_energy

    # (c): Calculate the average power level during a cooking event
    df_only_events['power_mean'] = df_only_events.event_energy / \
        (df_only_events.cooking_time / 60)

    # (d): Calculate the cost of cooking (Tanzanian Shilling)
    df_only_events['cooking_cost'] = df_only_events.event_energy * TZS_per_kWh

    # (e): Conduct renumbering the cooking event count
    df_only_events.reset_index(inplace=True)
    df_only_events['event_count'] = 0
    df_only_events.loc[(df_only_events.cooking_event.diff()
                        != 0), 'event_count'] += 1
    df_only_events.event_count = df_only_events['event_count'].cumsum()

    # Set timestamp in index to facilitate plotting with timeseries on the
    # x-axis.
    df_only_events.set_index('timestamp', inplace=True)
    df_only_events.drop({'cooking_event'},axis=1,inplace=True)
    return df_only_events


def format_timestamp(df):
    """
    Format column 'timestamp' for plotting timeseries in x-axis.

    Parameters
    ----------
    df : a dataframe that either has timestamp in the index column or will
    replace current index and with the timestamp.

    Returns
    -------
    df : Return a Pandas DataFrame with an timeseries index of localized time

    """
    def format_df():
        # Update numbering in index
        df.reset_index(inplace=True)

        # Make sure the timestamp is in type datetime
        df.timestamp = pd.to_datetime(df.timestamp)
        df.timestamp = np.int64(df.timestamp)
        df.timestamp = pd.to_datetime(df.timestamp)

        # Localize timestamp
        df['timestamp'] = df['timestamp'].dt.tz_localize(
            'utc').dt.tz_convert(df['timezone_region'].astype(str)[0])
        df.set_index('timestamp', inplace=True)

    if 'timestamp' in df.columns:

        df.set_index('timestamp', inplace=True)
        format_df()
    else:
        format_df()

    return df


def preprocess_epc_data(df_raw, **kwargs):
    """
    Preform a sequence of functions to produce the complete preprocessing
    of the raw EPC data to create a dataframe that can be used to conduct data
    analysis on how the households have used their EPCs.

    Parameters
    ----------
    df_raw : dataframe that is used as source file.
    **kwargs : add any parameters that are part of subfunctions to see the
    effect of parameter changes.

    Returns
    -------
    df_epc : returns a Pandas DataFrame.

    """

    df_raw = resolve_spikes(df_raw, **kwargs)
    df_processed = extract_cooking_events(df_raw, **kwargs)
    df_epc = resolve_timestamps(df_processed, **kwargs)
    df_epc = add_event_end(df_epc, **kwargs)
    df_epc = add_event_start(df_epc, **kwargs)
    df_epc = extract_cooking_events(df_epc, **kwargs)
    return df_epc


def clean_data_set(df_epc):
    """
    Clean up the dataframe after the preprocessing steps have been conducted.
    This is done by renumbering the cooking events and deleting "unnecessary columns"
    that shouldn't be part of the final dataframe.

    Parameters
    ----------
    df_epc : dataframe that has gone through the desired preprocessing steps.

    Returns
    -------
    df_epc : returns a Pandas DataFrame.

    """

    df_epc.reset_index(inplace=True)

    # Do a renumbering of the cooking event count
    df_epc['event_count'] = 0
    df_epc.loc[(
        (df_epc.cooking_event.diff()
         != 0)
        & (df_epc.cooking_event.isnull() == 0)
    ), 'event_count'] += 1
    df_epc.event_count = df_epc['event_count'].cumsum()
    df_epc.loc[(
        (df_epc.cooking_event.isnull())
    ), 'event_count'] += np.nan
    
    # Create column 'event_energy' for kWh per event.
    df_epc['event_energy'] = df_epc['energy_end'] - df_epc['energy_start']
    
    # Delete "unnecessary" columns
    df_epc.drop(['diff_next_timestamp', 'diff_prev_timestamp',
                 'timestamp_load', 'load_count', 'load',
                 'cooking_start', 'cooking_end', 'cooking_event',
                 'time_start', 'energy_start', 'time_end',
                 'energy_end', 'region', 'id'], axis=1, inplace=True)

    # Set meter_number in index to optimize the release file format for
    # combining source files relating to smart meter numbers.
    df_epc.set_index('meter_number', inplace=True)
    return df_epc

#Uncomment the lines below to execute the code directly in this file
'''

# Explaining parameters and their default values.
"""
'time_resolution' : 5 minutes is the frequency with which the smart meter is recording.

't_between' : 15 minutes is the time between two cooking events.

'energy_error_margin' : 0.04 kWh is how much two measurements that are taken
more or less simultaneously [+/- 2 minutes] can differ. This parameter is used
to resolve timestamp issues.

'mininum_size_of_spikes' : 1 kWh is the threshold to qualify a jump in the
energy as a corrupt value.

'minimum_energy_per_cooking_event' : 0.05 kWh is the least amount of energy consumption
that can be classified as a cooking event.

'minimum_power_mean' : 0.05 kW is the minimum average power level that a
cooking event needs to have to be classified as a cooking event.

'minimum_event_current' : 0.5 A is the minimum current that the EPC needs to have
recorded to qualify a cooking event.

'maximum_cooking_time' : 300 minutes is the maximum length that an EPC is believed to be cooking
before it's not considered a normal EPC event.

'minimum_active_load' : 0.15 is the min. average ratio of the Max. Rated Capacity
that a possible cooking event needs to have to be classified as a cooking event.

'power_capacity': 1 kW is the rated power capacity of the EPC in this study.
"""

# Change these parameters to influence all preprocessing steps through **kwargs.
params = {'time_resolution': 5,
         't_between': 15,
         'energy_error_margin': 0.04,
         'mininum_size_of_spikes': 1,
         'minimum_energy_per_cooking_event': 0.05,
         'minimum_power_mean': 0.05,
         'minimum_event_current': 0.5,
         'maximum_cooking_time':300,
         'time_resolution': 5,
         'minimum_active_load': 0.15,
         'power_capacity': 1}

# Source file
df_raw = pd.read_csv('dataframe_raw_jan14.csv', sep=',')

# Outputs
df_epc = preprocess_epc_data(df_raw, **params)
df_only_events = create_only_event_df(df_epc)
df_epc = clean_data_set(df_epc)
'''
