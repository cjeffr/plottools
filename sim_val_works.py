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


def fetch_noaa_data(station, begin_date, end_date, product, datum='MSL', time_zone='GMT'):
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


def compare_obs_sim(obs, sim_list, meta, product, savepath='./'):
    """
    Plotting routine for plotting NOAA gauge data vs. one or more simulation datasets.

    Parameters:
    - obs: DataFrame of observational data
    - sim_list: list of DataFrames for simulation data (can be length 1 or more)
    - meta: dict with station metadata
    - product: str, either 'currents' or 'water_level'
    - savepath: optional save path
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('mystyle.mplstyle')
    station = meta['id']

    # === Observational data ===
    if product == 'currents':
        if 'speed' in obs.columns:
            obs_data = obs['speed'].rolling(window=10, center=True).mean()
            obs_data /= 100  # Convert to m/s
        else:
            obs_data = np.sqrt(obs['u-vel']**2 + obs['v-vel']**2)
        ylabel = 'Current speed (m/s)'
    else:
        obs_data = obs['sea_surface'] + 0.161
        ylabel = 'Water Elevation (m)'

    # === Title ===
    if meta['name'] == 'null':
        title = f'Station {station}\n ({meta["lon"]}, {meta["lat"]})'
    else:
        title = f'Station {station}\n {meta["name"]}\n ({meta["lon"]}, {meta["lat"]})'

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(obs.index, obs_data, label='Observed', linewidth=2)

    for label, sim in sim_list:
        if product == 'currents':
            sim_data = np.sqrt(sim['u-vel']**2 + sim['v-vel']**2)
        else:
            sim_data = sim['zeta']
        # label = f'Simulation {idx+1}' if len(sim_list) > 1 else 'Simulation'
        ax.plot(sim.index, sim_data, label=label, linestyle='--')

    # === Axes formatting ===
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.set_ylabel(ylabel)

    # Y-axis minor ticks
    ymin, ymax = ax.get_ylim()
    minor_ticks = np.arange(np.floor(ymin * 4) / 4, np.ceil(ymax * 4) / 4 + 0.25, 0.25)
    ax.set_yticks(minor_ticks, minor=True)

    fig.suptitle(title)
    ax.legend(loc='upper left')

    if product == 'currents':
        ax.set_ylim(-0.05, 1.5)
    else:
        ax.set_ylim(-2, 2)

    # Optional: Save or show
    plt.tight_layout()
    plt.savefig(f"{savepath}/station_{station}_{product}.png", dpi=150)
    plt.show()

              
    fig.subplots_adjust(top=0.82)  # Reserve consistent space for title
    plt.savefig(f'tide_comparison_{station}.png')
    plt.show()




def compare_obs_sim_validation(
    obs, sim, meta, product,
    ignore_period=None,
    start_date=None,
    end_date=None,
    savepath='./'
):

    """
    Validation-style plot comparing observations and simulation with metrics and scatter plot.

    Parameters:
    - obs: DataFrame with observation data
    - sim: DataFrame with simulation data
    - meta: Dict with metadata (id, name, lon, lat)
    - product: 'currents' or 'water_level'
    - ignore_period: tuple of (start, end) datetime to exclude (e.g., storm)
    - savepath: Output directory
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score
    plt.style.use('mystyle.mplstyle')

        # === Slice data to user-defined period ===
    if start_date:
        obs = obs[obs.index >= pd.to_datetime(start_date)]
        sim = sim[sim.index >= pd.to_datetime(start_date)]
    if end_date:
        obs = obs[obs.index <= pd.to_datetime(end_date)]
        sim = sim[sim.index <= pd.to_datetime(end_date)]

    
    # === Extract and align data ===
    if product == 'currents':
        obs_data = obs['speed'].rolling(window=10, center=True).mean()
        obs_data /= 100  # Convert to m/s
        sim_data = np.sqrt(sim['u-vel']**2 + sim['v-vel']**2)
        ylabel = 'Current Speed (m/s)'
    else:
        obs_data = obs['sea_surface'] + 0.161
        sim_data = sim['zeta']
        ylabel = 'Water Level (m, NAVD88)'
    

    # === Align time indexes robustly ===
    # Use nearest neighbor reindexing within a small tolerance
    sim_data_aligned = sim_data.reindex(obs_data.index, method='nearest', tolerance=pd.Timedelta('10min'))
    
    # Combine into DataFrame
    combined = pd.DataFrame({'obs': obs_data, 'sim': sim_data_aligned})
    combined.dropna(inplace=True)
    
    # === Exclude ignore period, if any ===
    if ignore_period:
        mask = (combined.index >= ignore_period[0]) & (combined.index <= ignore_period[1])
        combined = combined[~mask]
    
    # === Sanity check before metric calculations ===
    if len(combined) == 0:
        raise ValueError("No overlapping data between observation and simulation after alignment.")
    # === Metrics ===
    rmse = np.sqrt(mean_squared_error(combined['obs'], combined['sim']))
    bias = (combined['sim'] - combined['obs']).mean()
    std_dev = (combined['sim'] - combined['obs']).std()
    r2 = r2_score(combined['obs'], combined['sim'])
    count = len(combined)

    # === Timing error metrics (optional placeholders) ===
    # For real timing metrics, you'd need peak detection. We'll skip for now.
    timing_avg = 1.4
    timing_min = -4.9
    timing_max = 7.5

    # === Plotting ===
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2)

    # --- 1. Time series ---
    ax1 = fig.add_subplot(gs[0:2, :])
    ax1.plot(combined.index, combined['obs'], label='Observed', linewidth=1.5)
    ax1.plot(combined.index, combined['sim'], label='Modeled', linestyle='--', linewidth=1.5)

    if ignore_period:
        ax1.axvspan(ignore_period[0], ignore_period[1], color='gray', alpha=0.3, label='Ignored Period')

    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('Date and Time')
    ax1.legend()
    ax1.set_title(f"Station {meta['id']} - {meta.get('name', '')}")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    ax1.grid(True)

    # --- 2. Metrics table ---
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.axis('off')
    table_data = [
        ['Count', f'{count}'],
        ['R²', f'{r2:.3f}'],
        ['RMSE', f'{rmse:.2f}'],
        ['Average Difference', f'{bias:.2f}'],
        ['Standard Deviation', f'{std_dev:.2f}'],
    ]
    ax2.table(cellText=table_data, colWidths=[0.5, 0.5], loc='center')

    # --- 3. Scatter plot ---
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.scatter(combined['sim'], combined['obs'], alpha=0.4, edgecolor='k', linewidth=0.3)
    min_val = min(combined.min())
    max_val = max(combined.max())
    ax3.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')  # 1:1 line
    ax3.set_xlabel('Modeled')
    ax3.set_ylabel('Observed')
    ax3.set_title('Observed vs Modeled')

    # --- Save ---
    plt.tight_layout()
    filename = f"{savepath}/validation_station_{meta['id']}_{product}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Validation figure saved to {filename}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Week
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.gridspec import GridSpec

def plot_weekly_validation(obs, sim, meta, product, start_date=None, num_weeks=4, title=None):
    plt.style.use('mystyle.mplstyle')
    station = meta['id']

    # === Title ===
    if meta['name'] == 'null':
        title = f'Station {station}\n ({meta["lon"]}, {meta["lat"]})'
    else:
        title = f'Station {station}\n {meta["name"]}\n ({meta["lon"]}, {meta["lat"]})'

        
    # === Extract and align data ===
    if product == 'currents':
        obs_data = obs['speed'].rolling(window=10, center=True).mean()
        obs_data /= 100  # Convert to m/s
        sim_data = np.sqrt(sim['u-vel']**2 + sim['v-vel']**2)
        ylabel = 'Current Speed (m/s)'
    else:
        obs_data = obs['sea_surface'] + 0.161
        sim_data = sim['zeta']
        ylabel = 'Water Level (m, NAVD88)'
        
    
    sim_data_aligned = sim_data.reindex(obs_data.index, method='nearest', tolerance=pd.Timedelta('10min'))

    combined = pd.DataFrame({'obs': obs_data, 'sim': sim_data_aligned}).dropna()
    
    if combined.empty:
        print("No overlapping data between observation and simulation.")
        return

    # === Time setup ===
    if start_date is None:
        start = combined.index.min().normalize()
    else:
        start = pd.to_datetime(start_date).normalize()
    weeks = [start + i * Week(1) for i in range(num_weeks + 1)]

    # === Layout setup ===
    ncols = 2
    nrows = (num_weeks + 1) // 2  # integer division for weekly plot rows
    total_rows = nrows + 1  # +1 for summary stats
    fig = plt.figure(figsize=(14, 3.5 * total_rows))
    gs = GridSpec(total_rows, ncols, figure=fig, height_ratios=[1]*nrows + [0.8])

    # === Weekly time series plots ===
    for i in range(num_weeks):
        wk_start, wk_end = weeks[i], weeks[i + 1]
        obs_wk = combined['obs'][wk_start:wk_end]
        sim_wk = combined['sim'][wk_start:wk_end]

        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.plot(obs_wk.index, obs_wk.values, label='Observed', linewidth=2)
        ax.plot(sim_wk.index, sim_wk.values, label='Simulated', linestyle='--')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
        ax.set_title(f"Week {i+1}: {wk_start.date()} to {wk_end.date()}")
        ax.grid(True)
        ax.legend()

    # === Metrics summary ===
    rmse = np.sqrt(mean_squared_error(combined['obs'], combined['sim']))
    bias = (combined['sim'] - combined['obs']).mean()
    std_dev = (combined['sim'] - combined['obs']).std()
    r2 = r2_score(combined['obs'], combined['sim'])
    count = len(combined)

    # --- Metrics Table ---
    ax_table = fig.add_subplot(gs[-1, 0])
    ax_table.axis('off')
    table_data = [
        ['Count', f'{count}'],
        ['R²', f'{r2:.3f}'],
        ['RMSE', f'{rmse:.2f}'],
        ['Average Difference', f'{bias:.2f}'],
        ['Standard Deviation', f'{std_dev:.2f}'],
    ]
    table = ax_table.table(cellText=table_data, colWidths=[0.5, 0.5], loc='center')
    table.scale(1, 1.5)  # Adjust horizontal (1) and vertical (1.5) scaling


    # --- Scatter Plot ---
    ax_scatter = fig.add_subplot(gs[-1, 1])
    ax_scatter.scatter(combined['sim'], combined['obs'], alpha=0.4, edgecolor='k', linewidth=0.3)
    min_val = min(combined.min())
    max_val = max(combined.max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--')
    ax_scatter.set_xlabel('Simulated')
    ax_scatter.set_ylabel('Observed')
    ax_scatter.set_title('Observed vs Simulated')

    # === Final layout ===
    if title:
        fig.suptitle(title, fontsize=20, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f'weekly_tide_comparison_{station}.png')
    plt.show()
