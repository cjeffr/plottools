import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime, timedelta
from submesh import io  # Assuming this reads fort.63/64.nc
from shapely.geometry import Point
from scipy.spatial import cKDTree
import requests
import io as stringio  # Add this import at the top

import io as stringio
import pandas as pd
import requests


def fetch_noaa_data(station, begin_date, end_date, product, datum='MSL', time_zone='LST'):
    import requests
    # NOAA API endpoint
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    # Define parameters
    params = {
        "product": product,
        "application": "NOS.COOPS.TAC.WL",
        "begin_date": begin_date,
        "end_date":   end_date,
        "datum": datum,
        "station": station,
        "time_zone": time_zone,
        "units": "metric",
        "format": "json"
    }
    # Make the request
    response = requests.get(url, params=params)
    # Check response status
    if response.status_code == 200:
        resp = response.json()
        data = resp.get("data", [])  # Get the list of readings
        meta = resp.get('metadata',[])
        df = pd.DataFrame(data)
        df = parse_noaa_data(df)
        return df, meta   # return as a  DataFrame
    else:
        raise requests.HTTPError(f"Request failed with status code {response.status_code}: {response.text}")

def parse_noaa_data(df):
    """Parse NOAA data read in from API
    Returns DataFrame with datetime index and data columns
    """
    # Works for water_level data
    cols = {'t': 'time', 's': 'speed', 'd': 'direction', 'v': 'sea_surface'}
    cols_to_keep = [v for k,v in cols.items()]
    df.t = pd.to_datetime(df.t)
    df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
    df = df[[col for col in df.columns if col in cols_to_keep]]
    df.set_index('time', inplace=True)
    df = df.apply(pd.to_numeric)
    return df

def select_simulation_data(ds, meta, variable):
    lon = float(meta['lon'])
    lat = float(meta['lat'])
    node = find_nearest_node(ds, lon, lat)
    return ds[variable].sel(node=node).to_dataframe()


def find_nearest_node(ds: xr.Dataset, lon, lat):
    """Find nearest ADCIRC node index given lat/lon."""
    nodes = np.column_stack((ds['x'].values, ds['y'].values))
    tree = cKDTree(nodes)
    _, idx = tree.query([lon, lat])
    return idx


def compare_obs_sim(obs, sim, meta, product, savepath='./'):
    """
    Plotting routine for plotting noaa gauge data vs. Simulation data:
    Datatype = 'current', 'water_level'
    These will change the plot configurations depending
    """
    import matplotlib.dates as mdates
    plt.style.use('mystyle.mplstyle')
    station = meta['id']
    if product == 'currents':
        # Smooth out current data from 3m to 1hr:
        obs_data = obs['speed'].rolling(window=10, center=True).mean()
        obs_data /= 100
        sim_data = np.sqrt(sim['u-vel']**2 + sim['v-vel']**2)
        ylabel = 'Current speed (m/s)'
    else:
        obs_data = obs['sea_surface']
        sim_data = sim.zeta
        ylabel = 'Water Elevation (m)'
    if meta['name'] == 'null':
        title = f'Station {station}\n ({meta['lon']}, {meta['lat']})'
    else:
        title = f'Station {station}\n {meta['name']}\n ({meta['lon']}, {meta['lat']})'
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(obs.index, obs_data, label='Observed')
    ax.plot(sim_data.index, sim_data, label='Simulation', linestyle='--')

    # Adjust axes
    ax.set_xlabel("Date")
    # Major ticks: 1 per day
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    # Minor ticks: every 6 hours
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    
    # === Y-axis formatting ===
    
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    # Minor ticks every 0.25 m
    minor_ticks = np.arange(np.floor(ymin * 4) / 4, np.ceil(ymax * 4) / 4 + 0.25, 0.25)
    ax.set_yticks(minor_ticks, minor=True)
    
    fig.suptitle(title) 
    ax.legend(loc='upper left')
    ax.set_ylim(-0.05, 1.5)
              
    fig.subplots_adjust(top=0.82)  # Reserve consistent space for title
    plt.savefig(f'tide_comparison_{station}.png')
    plt.show()