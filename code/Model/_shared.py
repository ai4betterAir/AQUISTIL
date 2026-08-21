import logging
import os

import pandas as pd


class ImputationBase:
    """Compatibility base used by the legacy class-based imputers."""

    def __init__(self, Configuration=None, logger=None, justif=80):
        self.Configuration = Configuration or type("Configuration", (), {})()
        self.logger = logger or logging.getLogger(__name__)
        self.justif = justif

    def extract_station_data(self, input_data_pd, var_to_predict=None):
        if input_data_pd is None or input_data_pd.empty:
            return {"ALL": input_data_pd.copy()}

        if not isinstance(input_data_pd, pd.DataFrame):
            input_data_pd = pd.DataFrame(input_data_pd)

        station_col = self._find_station_column(input_data_pd)
        if station_col is None:
            return {"ALL": input_data_pd.copy()}

        station_data = {}
        for station, group in input_data_pd.groupby(station_col, sort=False):
            station_key = str(station) if pd.notna(station) else "UNKNOWN"
            station_data[station_key] = group.copy()
        return station_data or {"ALL": input_data_pd.copy()}

    def combine_station_data(self, station_imputed_dict):
        frames = [df for df in station_imputed_dict.values() if isinstance(df, pd.DataFrame)]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=0)
        try:
            return combined.sort_index()
        except Exception:
            return combined

    def save_imputed_data(self, imputed_data_pd, station_name=None):
        output_dir = getattr(self.Configuration, "output_dir", None)
        if not output_dir:
            self.logger.debug("No output_dir configured; skipping legacy imputed data save")
            return None

        os.makedirs(output_dir, exist_ok=True)
        suffix = str(station_name).replace(os.sep, "_") if station_name else "combined"
        path = os.path.join(output_dir, f"imputed_{suffix}.csv")
        imputed_data_pd.to_csv(path, index=True)
        return path

    @staticmethod
    def _find_station_column(df):
        candidates = [
            "station",
            "Station",
            "site",
            "Site",
            "site_name",
            "SiteName",
            "station_name",
            "StationName",
        ]
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
