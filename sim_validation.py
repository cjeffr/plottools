import os
import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from submesh import io


@dataclass
class StationMetadata:
    """Data class to hold station metadata"""
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    depth: Optional[float] = None


def fetch_noaa_data(station_id: str, begin_date: str, end_date: str, product: str) -> dict:
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'application': 'NOS.COOPS.TAC.WL',
        'datum': "MSL",
        'time_zone': 'LST',
        'units': 'metric',
        'format': 'json',
        'begin_date': begin_date,
        'end_date': end_date,
        'station': station_id,
        'product': product
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def parse_noaa_data(json_data: dict, product: str) -> pd.Series:
    raw = json_data.get('data', [])
    if not raw:
        raise ValueError('No data found in NOAA response.')
    df = pd.DataFrame(raw)
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)

    if product == 'water_level':
        return df['v'].astype(float).rename('sea_surface')
    elif product =='currents':
        return df['s'].astype(float).rename('speed')
    else:
        raise ValueError(f'Unsupported product: {product}')
    
def extract_metadata(json_data: dict, station_id: str, depth: Optional[float] = None) -> StationMetadata:
    meta = json_data.get('metadata': {})
    return StationMetadata(
        station_id=station_id,
        station_name=meta.get('name', 'Unknown'),
        latitude=float(meta.get('lat',0)),
        longitude=float(meta.get('lon', 0)),
        depth=depth
    )

def find_closest_node(sim_ds: xr.Dataset, lon:float, lat:float) -> int:
    dist = np.sqrt((sim_ds.x - lon)**2 + (sim_ds.y - lat)**2)
    return dist.argmin().values

def extract_sim_data(sim_ds: xr.Dataset, node_idx: int, var: str) -> xr.DataArray:
    return sim_ds[var].isel(node=node_idx)

def plot_comparison(obs: pd.Series, sim: xr.DataArray, station_name: str, data_type:str,
                    save_path: Optional[str] = None):
    sim_time = pd.to_datetime(sim.time.values)
    sim_vals = sim.values

    obs_interp = obs.reindex(sim_time, method='nearest', tolerance='5min').dropna()
    valid_mask = np.isin(sim_time, obs_interp.index)
    sim_vals = sim_vals[valid_mask]
    sim_time = sim_time[valid_mask]
    
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(obs_interp.index, obs_interp.values, label='Observed', color='black', lw=2)
    ax.plot(sim_time, sim_vals, label='Simulated', color='blue', lw=2, linestyle='--')

    ylabel = 'Water Level (m)' if data_type == 'water_level' else 'Current Speed (m/s)'
    ax.set_title(f'{station_name} - {ylabel}')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        plt.show()

def process_station(
        station_id: str,
        config: Dict,
        sim_ds: xr.Dataset,
        begin_date: str,
        end_date: str,
        out_dir: Optional[str] = None):
    product = config['type']
    name = config.get('name', station_id)
    depth = config.get('depth')
    print(f'\n Processing {name} ({station_id}) [{product}]...')
    json_data = fetch_noaa_data(station_id, begin_date, end_date, product)
    obs = parse_noaa_data(json_data, product)
    meta = extract_metadata(json_data, station_id, depth)

    node_idx = find_closest_node(sim_ds, meta.longitude, meta.latitude)
    sim_var = 'zeta' if product == 'water_level' else ['u-vel', 'v-vel']
    sim = extract_sim_data(sim_ds, node_idx, sim_var)

    plot_path = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, f'{station_id}_{product}.png')
    plot_comparison(obs, sim, meta.station_name, product, save_path=plot_path)

def process_all_gauges(
        gauge_config: Dict[str, Dict],
        elev_ds: xr.Dataset,
        vel_ds: xr.Dataset,
        begin_date: str,
        end_date: str,
        out_dir: Optional[str] = None
):
    for sid, cfg in gauge_config.items():
        try:
            sim_ds = elev_ds if cfg['type'] == 'water_level' else vel_ds
            process_station(sid, cfg, sim_ds, begin_date, end_date, out_dir)
        except Exception as e:
            print(f' Failed to process {sid}: {e}')

def build_gauge_config(currents: Dict, tides: Dict) -> Dict[str, Dict]:
    gauges = {}
    
    for sid, info in currents.items():
        lon, lat = map(dms_to_decimal, info['location'])
        gauges[sid] = {
            'type': 'currents',
            'latitude': lat,
            'longitude': lon,
            'depth': info.get('depth'),
            'name': sid
        }
    
    for sid, info in tides.items():
        lon, lat = map(dms_to_decimal, info['location'])
        gauges[sid] = {
            'type': 'water_level',
            'latitude': lat,
            'longitude': lon,
            'name': sid
        }

    return gauges


# if __name__ == '__main__':
#     elev_ds = io.read_fort63_nc('/path/to/fort.63.nc')
#     vel_ds = io.read_fort63_nc('/path/to/fort.64.nc')

# class NOAADataParser:
#     """Handles parsing and cleaning of NOAA API response data"""
    
#     def __init__(self):
#         self.column_mapping = {
#             't': 'time', 
#             's': 'speed', 
#             'd': 'direction', 
#             'v': 'sea_surface'
#         }
    
#     def parse_raw_data(self, raw_data: list) -> pd.DataFrame:
#         """
#         Parse raw NOAA data into a clean DataFrame
        
#         Parameters:
#         -----------
#         raw_data : list
#             Raw data from NOAA API response
            
#         Returns:
#         --------
#         pd.DataFrame : Cleaned DataFrame with datetime index
#         """
#         if not raw_data:
#             raise ValueError("No data provided to parse")
            
#         df = pd.DataFrame(raw_data)
#         return self._clean_dataframe(df)
    
#     def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and standardize the DataFrame"""
#         # Convert time column to datetime
#         if 't' in df.columns:
#             df['t'] = pd.to_datetime(df['t'])
        
#         # Rename columns using mapping
#         df = df.rename(columns={k: v for k, v in self.column_mapping.items() 
#                                if k in df.columns})
        
#         # Keep only mapped columns that exist
#         valid_columns = [col for col in df.columns 
#                         if col in self.column_mapping.values()]
#         df = df[valid_columns]
        
#         # Set time as index if it exists
#         if 'time' in df.columns:
#             df.set_index('time', inplace=True)
            
#         return df


# class NOAAAPIClient:
#     """Handles communication with NOAA API"""
    
#     def __init__(self, base_url: str = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"):
#         self.base_url = base_url
#         self.default_params = {
#             "application": "NOS.COOPS.TAC.WL",
#             "datum": "MSL",
#             "time_zone": "LST",
#             "units": "metric",
#             "format": "json"
#         }
    
#     def fetch_data(self, begin_date: str, end_date: str, station_id: str, 
#                    datatype: str) -> Dict[str, Any]:
#         """
#         Fetch data from NOAA API
        
#         Parameters:
#         -----------
#         begin_date : str
#             Start date in YYYYMMDD format
#         end_date : str
#             End date in YYYYMMDD format
#         station_id : str
#             NOAA station number
#         datatype : str
#             Type of data ('water_level', 'currents', etc.)
            
#         Returns:
#         --------
#         dict : Raw API response data
#         """
#         params = {
#             **self.default_params,
#             "product": datatype,
#             "begin_date": begin_date,
#             "end_date": end_date,
#             "station": station_id
#         }
        
#         response = requests.get(self.base_url, params=params)
        
#         if response.status_code != 200:
#             raise requests.HTTPError(
#                 f"Request failed with status code {response.status_code}: {response.text}"
#             )
        
#         return response.json()


# class MetadataExtractor:
#     """Extracts and structures metadata from API responses"""
    
#     @staticmethod
#     def extract_station_metadata(api_response: Dict[str, Any], 
#                                 station_id: str, depth: Optional[float] = None) -> StationMetadata:
#         """
#         Extract station metadata from API response
        
#         Parameters:
#         -----------
#         api_response : dict
#             Raw API response
#         station_id : str
#             Station identifier
#         depth : float, optional
#             Depth of measurement
            
#         Returns:
#         --------
#         StationMetadata : Structured metadata
#         """
#         metadata = api_response.get("metadata", {})
        
#         return StationMetadata(
#             station_id=station_id,
#             station_name=metadata.get("name", "Unknown Station"),
#             latitude=float(metadata.get("lat", 0)),
#             longitude=float(metadata.get("lon", 0)),
#             depth=depth
#         )


# class SimulationComparator(ABC):
#     """Abstract base class for comparing observational data with simulation data"""
    
#     def __init__(self, sim_dataset: xr.Dataset):
#         self.sim_dataset = sim_dataset
    
#     def find_closest_node(self, lon: float, lat: float) -> Tuple[int, str]:
#         """Find the closest simulation node to given coordinates"""
#         distance = np.sqrt((self.sim_dataset.x - lon)**2 + (self.sim_dataset.y - lat)**2)
#         closest_idx = distance.argmin().values
#         return closest_idx, f"Node {closest_idx}"
    
#     @abstractmethod
#     def extract_simulation_data(self, node_idx: int) -> Any:
#         """Extract relevant simulation data for comparison"""
#         pass
    
#     @abstractmethod
#     def extract_observed_data(self, obs_df: pd.DataFrame) -> pd.Series:
#         """Extract relevant observed data for comparison"""
#         pass
    
#     def compare(self, obs_df: pd.DataFrame, metadata: StationMetadata) -> ComparisonResult:
#         """
#         Compare observed and simulated data
        
#         Parameters:
#         -----------
#         obs_df : pd.DataFrame
#             Observed data
#         metadata : StationMetadata
#             Station metadata
            
#         Returns:
#         --------
#         ComparisonResult : Comparison results
#         """
#         # Find closest simulation node
#         closest_idx, node_info = self.find_closest_node(metadata.longitude, metadata.latitude)
        
#         # Extract data
#         obs_data = self.extract_observed_data(obs_df)
#         sim_data = self.extract_simulation_data(closest_idx)
        
#         return ComparisonResult(
#             observed_data=obs_data,
#             observed_time=obs_df.index,
#             simulated_data=sim_data,
#             simulated_time=sim_data.time if hasattr(sim_data, 'time') else None,
#             metadata=metadata,
#             closest_node_info=node_info
#         )


# class WaterLevelComparator(SimulationComparator):
#     """Specific comparator for water level data"""
    
#     def extract_simulation_data(self, node_idx: int):
#         """Extract water level (zeta) from simulation"""
#         return self.sim_dataset.isel(node=node_idx).zeta
    
#     def extract_observed_data(self, obs_df: pd.DataFrame) -> pd.Series:
#         """Extract sea surface data from observations"""
#         if 'sea_surface' not in obs_df.columns:
#             raise ValueError("Sea surface data not available in observed data")
#         return obs_df['sea_surface']


# class CurrentComparator(SimulationComparator):
#     """Specific comparator for current data"""
    
#     def extract_simulation_data(self, node_idx: int):
#         """Extract current speed from simulation u and v components"""
#         sim_data = self.sim_dataset.isel(node=node_idx)
#         return np.sqrt(sim_data['u-vel']**2 + sim_data['v-vel']**2)
    
#     def extract_observed_data(self, obs_df: pd.DataFrame) -> pd.Series:
#         """Extract current speed from observations"""
#         if 'speed' not in obs_df.columns:
#             raise ValueError("Current speed data not available in observed data")
#         return obs_df['speed']


# class NOAADataProcessor:
#     """Main class that orchestrates the entire data processing workflow"""
    
#     def __init__(self, api_client: Optional[NOAAAPIClient] = None, 
#                  parser: Optional[NOAADataParser] = None):
#         self.api_client = api_client or NOAAAPIClient()
#         self.parser = parser or NOAADataParser()
#         self.metadata_extractor = MetadataExtractor()
    
#     def process_data(self, begin_date: str, end_date: str, station_id: str, 
#                     datatype: str, depth: Optional[float] = None) -> Tuple[pd.DataFrame, StationMetadata]:
#         """
#         Process NOAA data from API to clean DataFrame with metadata
        
#         Parameters:
#         -----------
#         begin_date : str
#             Start date in YYYYMMDD format
#         end_date : str
#             End date in YYYYMMDD format
#         station_id : str
#             NOAA station number
#         datatype : str
#             Type of data ('water_level', 'currents', etc.)
#         depth : float, optional
#             Depth of measurement
            
#         Returns:
#         --------
#         tuple : (cleaned DataFrame, station metadata)
#         """
#         # Fetch raw data from API
#         raw_response = self.api_client.fetch_data(begin_date, end_date, station_id, datatype)
        
#         # Extract metadata
#         metadata = self.metadata_extractor.extract_station_metadata(
#             raw_response, station_id, depth
#         )
        
#         # Parse and clean data
#         raw_data = raw_response.get("data", [])
#         if not raw_data:
#             raise ValueError("No data returned for the specified parameters.")
        
#         cleaned_df = self.parser.parse_raw_data(raw_data)
        
#         return cleaned_df, metadata
    
#     def compare_with_simulation(self, obs_df: pd.DataFrame, metadata: StationMetadata,
#                               sim_dataset: xr.Dataset, datatype: str) -> ComparisonResult:
#         """
#         Compare observational data with simulation
        
#         Parameters:
#         -----------
#         obs_df : pd.DataFrame
#             Observational data
#         metadata : StationMetadata
#             Station metadata
#         sim_dataset : xarray Dataset
#             Simulation dataset
#         datatype : str
#             Type of data being compared
            
#         Returns:
#         --------
#         ComparisonResult : Comparison results
#         """
#         # Create appropriate comparator based on data type
#         if datatype == 'water_level':
#             comparator = WaterLevelComparator(sim_dataset)
#         elif datatype == 'currents':
#             comparator = CurrentComparator(sim_dataset)
#         else:
#             raise ValueError(f"Unsupported datatype: {datatype}")
        
#         return comparator.compare(obs_df, metadata)
    
#     def process_station_complete(self, begin_date: str, end_date: str, station_id: str, 
#                                datatype: str, sim_dataset: xr.Dataset, 
#                                depth: Optional[float] = None) -> ComparisonResult:
#         """
#         Complete workflow: pull data, clean, and prepare comparison for a single station
        
#         Parameters:
#         -----------
#         begin_date, end_date : str
#             Date range in YYYYMMDD format
#         station_id : str
#             NOAA station number
#         datatype : str
#             'water_level' or 'currents'
#         sim_dataset : xarray Dataset
#             Simulation data (elev or vel)
#         depth : float, optional
#             Measurement depth
            
#         Returns:
#         --------
#         ComparisonResult : Ready-to-plot comparison data
#         """
#         # Pull and clean NOAA data
#         obs_df, metadata = self.process_data(begin_date, end_date, station_id, datatype, depth)
        
#         # Compare with simulation
#         comparison = self.compare_with_simulation(obs_df, metadata, sim_dataset, datatype)
        
#         return comparison


# class DataPlotter:
#     """Handles plotting of observational vs simulation data"""
    
#     def __init__(self):
#         import matplotlib.pyplot as plt
#         import matplotlib.dates as mdates
#         self.plt = plt
#         self.mdates = mdates
        
#         # Define plotting configurations by data type
#         self.plot_configs = {
#             'water_level': {
#                 'ylabel_base': 'Water Level',
#                 'default_units': 'm',
#                 'observed_color': 'black',
#                 'observed_style': {'linewidth': 2, 'marker': 'o', 'markersize': 3},
#                 'sim_color': 'blue',
#                 'sim_style': {'linestyle': '-.', 'linewidth': 2}
#             },
#             'currents': {
#                 'ylabel_base': 'Current Speed', 
#                 'default_units': 'm/s',
#                 'observed_color': 'red',
#                 'observed_style': {'linewidth': 2, 'alpha': 0.8},
#                 'sim_color': 'blue', 
#                 'sim_style': {'linewidth': 2, 'alpha': 0.8}
#             }
#         }
    
#     def _apply_smoothing(self, data: pd.Series, smoothing_config: Optional[Dict] = None) -> pd.Series:
#         """Apply smoothing to data if requested"""
#         if not smoothing_config:
#             return data
            
#         method = smoothing_config.get('method', 'rolling_mean')
#         window = smoothing_config.get('window', 10)
        
#         if method == 'rolling_mean':
#             return data.rolling(window=window, center=True).mean()
#         elif method == 'rolling_median':
#             return data.rolling(window=window, center=True).median()
#         elif method == 'gaussian':
#             try:
#                 from scipy.ndimage import gaussian_filter1d
#                 smoothed_data = data.copy()
#                 sigma = smoothing_config.get('sigma', 2)
#                 smoothed_data.values[:] = gaussian_filter1d(smoothed_data.values, sigma=sigma)
#                 return smoothed_data
#             except ImportError:
#                 print("Scipy not available, falling back to rolling mean")
#                 return data.rolling(window=window, center=True).mean()
#         else:
#             return data
    
#     def _convert_units(self, data: pd.Series, from_units: str, to_units: str) -> pd.Series:
#         """Convert data between units"""
#         conversion_factors = {
#             ('cm/s', 'm/s'): 0.01,
#             ('m/s', 'cm/s'): 100.0,
#             ('cm', 'm'): 0.01,
#             ('m', 'cm'): 100.0
#         }
        
#         factor = conversion_factors.get((from_units, to_units), 1.0)
#         return data * factor
    
#     def plot_comparison(self, observed_data: pd.Series, simulated_data, 
#                        datatype: str, station_name: str = "Station",
#                        observed_units: Optional[str] = None, 
#                        simulated_units: Optional[str] = None,
#                        smoothing_config: Optional[Dict] = None,
#                        show_plot: bool = True, save_path: Optional[str] = None,
#                        interpolate_to_sim: bool = True,
#                        tolerance: str = '4min') -> None:
#         """
#         Unified plotting routine for all data types and configurations
        
#         Parameters:
#         -----------
#         observed_data : pd.Series
#             Observational data with datetime index
#         simulated_data : xarray.DataArray or pd.Series
#             Simulation data with time coordinate/index
#         datatype : str
#             Type of data ('water_level' or 'currents') - determines styling
#         station_name : str
#             Name of the station for plot title
#         observed_units : str, optional
#             Units of observational data (if None, uses default for datatype)
#         simulated_units : str, optional  
#             Units of simulation data (if None, uses default for datatype)
#         smoothing_config : dict, optional
#             Smoothing configuration: {'method': 'rolling_mean', 'window': 10}
#             Methods: 'rolling_mean', 'rolling_median', 'gaussian'
#         show_plot : bool
#             Whether to display the plot
#         save_path : str, optional
#             Path to save the plot
#         interpolate_to_sim : bool
#             Whether to interpolate observed data to simulation timestamps
#         tolerance : str
#             Tolerance for interpolation (e.g., '4min')
#         """
#         # Get plot configuration for this data type
#         if datatype not in self.plot_configs:
#             raise ValueError(f"Unsupported datatype: {datatype}. Use: {list(self.plot_configs.keys())}")
        
#         config = self.plot_configs[datatype]
        
#         # Set default units if not provided
#         obs_units = observed_units or config['default_units']
#         sim_units = simulated_units or config['default_units']
        
#         # Apply smoothing if requested
#         obs_data_processed = self._apply_smoothing(observed_data, smoothing_config)
        
#         # Convert units to match simulation
#         if obs_units != sim_units:
#             obs_data_processed = self._convert_units(obs_data_processed, obs_units, sim_units)
        
#         # Handle simulation data (could be xarray or pandas)
#         if hasattr(simulated_data, 'time'):  # xarray DataArray
#             sim_times = pd.to_datetime(simulated_data.time.values)
#             sim_values = simulated_data.values
#         else:  # pandas Series
#             sim_times = simulated_data.index
#             sim_values = simulated_data.values
        
#         # Interpolate observed data to simulation timestamps if requested
#         if interpolate_to_sim:
#             obs_data_interp = obs_data_processed.reindex(
#                 sim_times, method='nearest', tolerance=pd.Timedelta(tolerance)
#             ).dropna()
            
#             # Filter simulation data to match interpolated observations
#             valid_mask = np.isin(sim_times, obs_data_interp.index)
#             sim_times_plot = sim_times[valid_mask]
#             sim_values_plot = sim_values[valid_mask]
#             obs_times_plot = obs_data_interp.index
#             obs_values_plot = obs_data_interp.values
#         else:
#             # Use data as-is
#             obs_times_plot = obs_data_processed.index
#             obs_values_plot = obs_data_processed.values
#             sim_times_plot = sim_times
#             sim_values_plot = sim_values
        
#         # Create plot
#         fig, ax = self.plt.subplots(figsize=(12, 6))
        
#         # Plot observed data
#         obs_label = 'NOAA Observed'
#         if smoothing_config:
#             obs_label += f" ({smoothing_config.get('method', 'smoothed')})"
        
#         ax.plot(obs_times_plot, obs_values_plot, 
#                 color=config['observed_color'], label=obs_label,
#                 **config['observed_style'])
        
#         # Plot simulation data  
#         ax.plot(sim_times_plot, sim_values_plot,
#                 color=config['sim_color'], label='Simulation',
#                 **config['sim_style'])
        
#         # Formatting
#         ylabel = f"{config['ylabel_base']} ({sim_units})"
#         title_suffix = " (Smoothed)" if smoothing_config else ""
        
#         ax.set_xlabel('Time', fontsize=12, fontweight='bold')
#         ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
#         ax.set_title(f'{station_name} - Observed vs Simulated {config["ylabel_base"]}{title_suffix}', 
#                     fontsize=14, fontweight='bold', pad=15)
#         ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
#         ax.grid(True, alpha=0.3, linestyle='--')
        
#         # Format dates on x-axis
#         ax.xaxis.set_major_formatter(self.mdates.DateFormatter('%m-%d %H:%M'))
#         ax.xaxis.set_major_locator(self.mdates.HourLocator(interval=12))
#         self.plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
#         self.plt.tight_layout()
        
#         if save_path:
#             self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
#         if show_plot:
#             self.plt.show()
#         else:
#             self.plt.close()


# class MultiGaugeProcessor:
#     """Processes multiple gauges in batch and generates plots"""
    
#     def __init__(self, processor: Optional[NOAADataProcessor] = None, 
#                  plotter: Optional[DataPlotter] = None):
#         self.processor = processor or NOAADataProcessor()
#         self.plotter = plotter or DataPlotter()
    
#     def process_all_gauges(self, begin_date: str, end_date: str, gauge_config: Dict[str, Dict],
#                           elev_dataset: xr.Dataset, vel_dataset: xr.Dataset,
#                           plot_results: bool = True, save_plots: bool = False,
#                           plot_directory: str = "./plots/") -> Dict[str, ComparisonResult]:
#         """
#         Process all gauges and optionally generate plots
        
#         Parameters:
#         -----------
#         begin_date, end_date : str
#             Date range in YYYYMMDD format
#         gauge_config : dict
#             Dictionary with gauge configurations. Format:
#             {
#                 'station_id': {
#                     'type': 'water_level' or 'currents',
#                     'depth': float (optional, for currents),
#                     'name': str (optional, for display)
#                 }
#             }
#         elev_dataset : xarray Dataset
#             Elevation/water level simulation data
#         vel_dataset : xarray Dataset  
#             Velocity/current simulation data
#         plot_results : bool
#             Whether to generate plots
#         save_plots : bool
#             Whether to save plots to files
#         plot_directory : str
#             Directory to save plots
            
#         Returns:
#         --------
#         dict : Results for all processed stations
#         """
#         import os
        
#         results = {}
        
#         # Create plot directory if saving plots
#         if save_plots and not os.path.exists(plot_directory):
#             os.makedirs(plot_directory)
        
#         for station_id, config in gauge_config.items():
#             try:
#                 print(f"Processing station {station_id}...")
                
#                 # Determine which simulation dataset to use
#                 if config['type'] == 'water_level':
#                     sim_dataset = elev_dataset
#                 elif config['type'] == 'currents':
#                     sim_dataset = vel_dataset
#                 else:
#                     print(f"Unknown data type {config['type']} for station {station_id}")
#                     continue
                
#                 # Process the station
#                 comparison = self.processor.process_station_complete(
#                     begin_date, end_date, station_id, config['type'], 
#                     sim_dataset, config.get('depth')
#                 )
                
#                 # Store results
#                 results[station_id] = comparison
                
#                 # Generate plots if requested
#                 if plot_results:
#                     station_name = comparison.metadata.station_name
#                     depth_info = f" (depth: {config.get('depth', 'unknown')}m)" if config.get('depth') else ""
#                     full_station_name = station_name + depth_info
                    
#                     # Determine data type for labeling
#                     data_type = "Water Level" if config['type'] == 'water_level' else "Current Speed"
#                     units = "m" if config['type'] == 'water_level' else "m/s"
                    
#                     # Prepare save path if needed
#                     save_path = None
#                     if save_plots:
#                         safe_name = station_id.replace('/', '_').replace('\\', '_')
#                         save_path = os.path.join(plot_directory, f'{safe_name}_{config["type"]}.png')
                    
#                     # Plot the comparison using the unified plotting method
#                     self.plotter.plot_comparison(
#                         observed_data=comparison.observed_data,
#                         simulated_data=comparison.simulated_data,
#                         datatype=config['type'],
#                         station_name=full_station_name,
#                         show_plot=plot_results and not save_plots,
#                         save_path=save_path
#                     )
                    
#                     print(f"Plotted {full_station_name}")
                
#                 # Optionally save data to CSV
#                 csv_path = f'{station_id}_{config["type"]}_data.csv'
#                 comparison.observed_data.to_csv(csv_path)
#                 print(f"Saved data to {csv_path}")
                
#             except Exception as e:
#                 print(f"Error processing station {station_id}: {e}")
#                 continue
        
# # Example usage with different plotting options:
# def example_plotting_usage():
#     """Examples of how to use the unified plotting system"""
    
#     # Basic plotting (assumes you have data)
#     plotter = DataPlotter()
    
#     # Example 1: Water level data with no smoothing
#     plotter.plot_comparison(
#         observed_data=water_level_series,
#         simulated_data=sim_water_level,
#         datatype='water_level',
#         station_name='Station 8679598'
#     )
    
#     # Example 2: Current data with smoothing
#     # plotter.plot_comparison(
#     #     observed_data=current_speed_series,
#     #     simulated_data=sim_current_speed,
#     #     datatype='currents', 
#     #     station_name='Kings Bay Pier',
#     #     smoothing_config={'method': 'rolling_mean', 'window': 10},
#     #     save_path='kingsbaypier_speed_smoothed.png'
#     # )
    
#     # Example 3: With unit conversion
#     # plotter.plot_comparison(
#     #     observed_data=current_speed_cm_per_s,
#     #     simulated_data=sim_current_m_per_s,
#     #     datatype='currents',
#     #     station_name='Station kb0102',
#     #     observed_units='cm/s',
#     #     simulated_units='m/s'
#     # )
    
#     # Example 4: Gaussian smoothing for noisy data
#     # plotter.plot_comparison(
#     #     observed_data=noisy_current_data,
#     #     simulated_data=sim_current_data,
#     #     datatype='currents',
#     #     station_name='Noisy Station',
#     #     smoothing_config={'method': 'gaussian', 'sigma': 2}
#     # )
    
#     print("Plotting examples ready - uncomment the ones you want to use!")


# # Backward compatibility functions (updated to use new plotting)
# def plot_gauge_data(gdata, sim_data, station_name="Station", data_type="Data", 
#                    gdata_units="cm/s", sim_units="m/s"):
#     """Backward compatible plotting function"""
#     plotter = DataPlotter()
    
#     # Determine datatype from data_type string
#     if 'water' in data_type.lower() or 'level' in data_type.lower():
#         datatype = 'water_level'
#     else:
#         datatype = 'currents'
    
#     plotter.plot_comparison(
#         observed_data=gdata,
#         simulated_data=sim_data,
#         datatype=datatype,
#         station_name=station_name,
#         observed_units=gdata_units,
#         simulated_units=sim_units
#     )


# # Convenience functions for backward compatibility
# def pull_and_process_NOAA_data(beginDate, endDate, stationNum, datatype, depth=None):
#     """Backward compatible function using the new OOP structure"""
#     processor = NOAADataProcessor()
#     obs_df, metadata = processor.process_data(beginDate, endDate, stationNum, datatype, depth)
    
#     return {
#         'data': obs_df,
#         'metadata': {
#             'station_id': metadata.station_id,
#             'station_name': metadata.station_name,
#             'latitude': metadata.latitude,
#             'longitude': metadata.longitude,
#             'depth': metadata.depth
#         }
#     }


# def process_all_gauges(beginDate, endDate, gauge_config, elev_dataset, vel_dataset):
#     """Backward compatible function for processing all gauges"""
#     multi_processor = MultiGaugeProcessor()
#     results = multi_processor.process_all_gauges(
#         beginDate, endDate, gauge_config, elev_dataset, vel_dataset
#     )
    
#     # Convert to old format for compatibility
#     converted_results = {}
#     for station_id, comparison in results.items():
#         converted_results[station_id] = {
#             'observed': {
#                 'time': comparison.observed_time,
#                 'values': comparison.observed_data,
#                 'metadata': {
#                     'station_id': comparison.metadata.station_id,
#                     'station_name': comparison.metadata.station_name,
#                     'latitude': comparison.metadata.latitude,
#                     'longitude': comparison.metadata.longitude,
#                     'depth': comparison.metadata.depth
#                 }
#             },
#             'simulated': {
#                 'time': comparison.simulated_time,
#                 'values': comparison.simulated_data,
#                 'location': comparison.closest_node_info
#             }
#         }
    
#     return converted_results