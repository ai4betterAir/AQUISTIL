"""
..  module:: Input_manager
    :platform: Unix
    :synopsis: Definition of the basic object class to output .

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

import sys
import os
import glob
import stat
import numpy as np
import pandas as pd
import pytz
import datetime as dtime
import itertools
import shutil

from . import File_input as FI
from . import aqms_api 
import Core_iHPC.Outputs.Output_manager as OM

###########################################################################################
class Input_manager_Class(object):
    """ 
    This class defines a Input_Class, that contains the capacity to manage the inputs.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, Configuration,
                ):

        self.Configuration = Configuration
        self.logger = self.Configuration.logger
        self.justif = self.Configuration.justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Configuring the Input manager'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))


        self.var_to_predict =  self.Configuration.var_to_predict
        self.additional_var_to_select = self.Configuration.additional_var_to_select

        self.selected_region = self.Configuration.selected_region

        self.Data_input_flow = self.Configuration.Data_input_flow

        self.Hubert_files_dir = "data/2018_2022"
        self.use_file = self.Configuration.use_file
        self.OBS_save_file_template = "allobs_{region}_{var}.csv"
        self.OBS_save_file_template = 'Allobs_{raw_processed}_{input_stream}_{region}_{var}_{additional_vars}.csv'

        self.aedt = pytz.timezone('Australia/Sydney')
        self.aest = pytz.timezone('Australia/Brisbane')

        self.threshold_missing_data = 0.5
        return

###########################################################################################
    def MakeDir(self, ddir):
        ''' This function makes the different working directories
        '''
        if not os.path.exists(ddir):
            os.makedirs(ddir)
            mod775 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IXUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IXGRP |stat.S_IROTH |stat.S_IXOTH
            os.chmod(ddir,mod775)
            #os.chmod(self.RunDir,0775)
            self.logger.info('Directory = {msg}'.format(msg=ddir).ljust(self.justif-7,'.') + 'CREATED')
        return    


# ###########################################################################################
    def clean_folder(self, folder_path):
        if os.path.exists(folder_path):
            print(f"The folder '{folder_path}' exists.")
        
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Deleted file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"Deleted directory: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        else:
            return

###########################################################################################
    def Change_permissions(self, path):
        ''' This function makes the different working directories
        '''
        import stat
        import os
        os.umask(0)
        mod664 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH  
        mod666 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IROTH |stat.S_IWOTH
        # os.chmod(path, mod664)
        os.chmod(path, mod666)
        return    

###########################################################################################
    def _normalize_variable_selection(self, variables):
        if variables is None:
            return []
        if isinstance(variables, str):
            return [item.strip() for item in variables.split(",") if item.strip()]
        normalized = []
        for item in variables:
            if item is None:
                continue
            if isinstance(item, str) and "," in item:
                normalized.extend([part.strip() for part in item.split(",") if part.strip()])
            else:
                normalized.append(item)
        return normalized

###########################################################################################
    def _resolve_variable_token(self, variable, mapping):
        if variable is None:
            return None

        token = str(variable).strip()
        if not token:
            return None

        if token in mapping:
            return token

        upper_to_key = {str(key).upper(): key for key in mapping.keys()}
        token_upper = token.upper()
        if token_upper in upper_to_key:
            return upper_to_key[token_upper]

        alias_map = {
            "TEMP": "T",
            "TEMPERATURE": "T",
            "T": "T",
            "OZONE": "O3",
            "O3": "O3",
            "PM25": "PM2.5",
            "PM_25": "PM2.5",
            "PM2_5": "PM2.5",
            "PM2.5": "PM2.5",
            "WIND": "Wind",
        }
        alias = alias_map.get(token_upper)
        if alias is not None:
            if alias in mapping:
                return alias
            alias_upper = alias.upper()
            if alias_upper in upper_to_key:
                return upper_to_key[alias_upper]

        return token

###########################################################################################
    def select_input_variables(self, additional_var_to_select):
        """
        method to only select the desired vars in to the dataframe
        """
        import itertools

        additional_var_to_select = self._normalize_variable_selection(additional_var_to_select)

        self.var_selection_dict = {
            "Wind" : ['Wind_U', 'Wind_V'],
            "CO" : ["CO"],
            "NO2" : ["NO2"],
            "PM10" : ["PM10"],
            "NEPH" : ["NEPH"],
            "CTM" : ["CTM"],
            "T" : ["T"],
            "TEMP" : ["TEMP"],
            "RH" : ["RH"],
            'RAIN' : ['RAIN'],
        }

        additional_var_to_select = [
            self._resolve_variable_token(variable, self.var_selection_dict)
            for variable in additional_var_to_select
        ]

        input_var_data_list = sorted(list(itertools.chain.from_iterable([self.var_selection_dict[ii] for ii in additional_var_to_select] )))
        # input_data_pd = pd.concat([df_other.loc[:,input_var_data], source_df], axis=1)
        # print(input_data_pd)

        ### update Configure
        self.Configuration.var_data_list_from_input_pd = input_var_data_list
        #

        self.logger.info("Additional variable selection".ljust(self.justif - 2 ,'.') + 'OK')
        return input_var_data_list  

###########################################################################################
    def load_dpe_api_input(self, input_stream):
        ''' 
        This function connects with the dpe api to get station data to feed the forecast dataflow.
        
        '''
        var_to_predict = self._normalize_variable_selection(self.Configuration.var_to_predict)
        additional_var_to_select = self._normalize_variable_selection(self.Configuration.additional_var_to_select)

        self.AQMS =  aqms_api.aqms_api_class()

        requested_station_names = sorted(set(
            self.Configuration.dpie_input_station_list + self.Configuration.dpie_output_station_list
        ))
        self.logger.info(
            "Region {region} station set: {stations}".format(
                region=self.Configuration.selected_region,
                stations=", ".join(requested_station_names),
            ).ljust(self.justif - 2, '.') + 'OK'
        )

        pd.set_option('display.max_rows', 500)
        ######################
        # get stations site details and make a dict out of it
        site_details_cache_dir = (
            self.Configuration.Main_model_data_dir
        )
        site_details_cache_path = os.path.join(site_details_cache_dir, "site_details_cache.csv")

        if os.path.isfile(site_details_cache_path):
            self.dpe_station_site_detail_pd = pd.read_csv(site_details_cache_path)
            self.logger.info(
                'Site details metadata'.ljust(self.justif - 2, '.') + 'LOADED FROM CACHE'
            )
        else:
            AllSites = self.AQMS.get_site_details()
            self.dpe_station_site_detail_pd = pd.json_normalize(AllSites.json())
            self.logger.info(
                'Site details metadata'.ljust(self.justif - 2, '.') + 'DOWNLOADED'
            )

        self.dpe_station_site_detail_pd["SiteName"] = (
            self.dpe_station_site_detail_pd.loc[:, "SiteName"]
            .astype(str)
            .str.strip()
            .str.replace(r"[\s\-]+", "_", regex=True)
        )
        self.dpe_station_site_detail_pd.set_index("SiteName", drop=True, inplace=True)
        # drop the duplicated station names
        self.dpe_station_site_detail_pd = self.dpe_station_site_detail_pd[~self.dpe_station_site_detail_pd.index.duplicated(keep='first')]
        if not os.path.isfile(site_details_cache_path):
            os.makedirs(site_details_cache_dir, exist_ok=True)
            self.dpe_station_site_detail_pd.reset_index().to_csv(site_details_cache_path, index=False)

        # print(self.dpe_station_site_detail_pd.sort_index(axis="index"))
        self.dpe_station_site_detail_dict = self.dpe_station_site_detail_pd.to_dict(orient='index')

        def _normalize_station_name(value):
            return str(value).strip().upper().replace("-", "_").replace(" ", "_")

        self.dpe_station_site_detail_lookup = {
            _normalize_station_name(name): name
            for name in self.dpe_station_site_detail_dict.keys()
        }
        
        self.dpe_reverse_station_site_detail_dict = {}
        for key,val in self.dpe_station_site_detail_dict.items():
            self.dpe_reverse_station_site_detail_dict[val["Site_Id"]] = key

        # print(self.dpe_station_site_detail_dict)
        # print(self.dpe_reverse_station_site_detail_dict)

        ######################
        # get stations parameters and make a dict out of it
        Allparameters = self.AQMS.get_parameters_details()
        self.dpe_station_parameters_pd = pd.json_normalize(Allparameters.json())
        # grab hourly values and hourly averages
        mask = (
            (self.dpe_station_parameters_pd["SubCategory"] == "Hourly") 
            & (self.dpe_station_parameters_pd["Frequency"] == "Hourly average")
            & (self.dpe_station_parameters_pd["Category"] == "Averages")
        )
        
        self.dpe_station_parameters_dict = self.dpe_station_parameters_pd[mask].set_index("ParameterCode", drop=True, inplace=False).to_dict(orient='index')
        # print(self.dpe_station_parameters_dict)

        dpe_species_properties_dict = {}
        for spec, val in self.dpe_station_parameters_dict.items():
            dpe_species_properties_dict[spec] = {
                'Units': val['Units'],
                'Species_name':val['ParameterDescription'],
                'Units_name':val['UnitsDescription'],
            }

        
        ######################
        # build variable list to get from api
        # correspondance dict between input and api var name necessary
        self.dpe_station_available_variable_list = self.dpe_station_parameters_dict.keys()
        # print(self.dpe_station_available_variable_list)
        
        # ['CO', 'HUMID', 'NEPH', 'NH3', 'NO', 'NO2', 'OZONE', 'PM10', 'PM10d', 'PM2.5', 'PM2.5d', 
        # 'RAIN', 'SD1', 'SO2', 'SOLAR', 'TEMP', 'TSPd', 'WDR', 'WSP']
        
        self.dpe_var_selection_dict = {
            "Wind" : ['WDR', 'WSP'],
            "CO" : ["CO"],
            "NO2" : ["NO2"],
            "PM10" : ["PM10"],
            "NEPH" : ["NEPH"],
            "CTM" : ["CTM"],
            "T" : ["TEMP"],
            "TEMP" : ["TEMP"],
            "RH" : ["RH"],
            'HUMID' :['HUMID'],
            'NH3' : ['NH3'],
            'NO' : ['NO'],
            "O3" : ['OZONE'],
            "OZONE" : ['OZONE'],
            'PM10d' : ['PM10d'],
            'PM2.5' : ['PM2.5'],
            'PM2.5d' : ['PM2.5d'],
            'RAIN' : ['RAIN'],
            'SD1' : ['SD1'],
            'SO2' : ['SO2'],
            'SOLAR' : ['SOLAR'],
            'TSPd' : ['TSPd'],
            }

        var_to_predict = [
            self._resolve_variable_token(variable, self.dpe_var_selection_dict)
            for variable in var_to_predict
        ]
        additional_var_to_select = [
            self._resolve_variable_token(variable, self.dpe_var_selection_dict)
            for variable in additional_var_to_select
        ]

        dpe_selected_var_list = sorted(list(
            itertools.chain.from_iterable(
                [self.dpe_var_selection_dict[ii] for ii in var_to_predict + additional_var_to_select] )
                ))

        # print(dpe_selected_var_list)

        ######################
        # build station list to get from api

        dpe_selected_station_id = []
        missing_stations = []
        for station_name in requested_station_names:
            resolved_station_name = self.dpe_station_site_detail_lookup.get(_normalize_station_name(station_name))
            if resolved_station_name is None:
                missing_stations.append(station_name)
                continue
            dpe_selected_station_id.append(self.dpe_station_site_detail_dict[resolved_station_name]["Site_Id"])

        if missing_stations:
            self.logger.warning(
                "Missing station names in API site details: {stations}".format(
                    stations=", ".join(sorted(set(missing_stations)))
                ).ljust(self.justif - 2, '.') + 'SKIP'
            )
        # make list unique
        dpe_selected_station_id = list(set(dpe_selected_station_id))

        # print(dpe_selected_station_id)

        ######################
        # Determine if we need to batch the downloads
        BATCH_SIZE = 4  # Download max 4 stations at a time
        total_stations = len(dpe_selected_station_id)
        
        if total_stations >= 5:
            # Split stations into batches
            station_batches = [dpe_selected_station_id[i:i + BATCH_SIZE] 
                             for i in range(0, total_stations, BATCH_SIZE)]
            self.logger.info(f'Large region detected ({total_stations} stations)'.ljust(self.justif,'.'))
            self.logger.info(f'Will download in {len(station_batches)} batches of {BATCH_SIZE} stations'.ljust(self.justif,'.'))
        else:
            # Single batch for small regions
            station_batches = [dpe_selected_station_id]
            
        ######################
        # build the base api request (common parameters)
        base_ObsRequest = self.AQMS.ObsRequest_init()
        base_ObsRequest['Categories'] = ["Averages"]
        base_ObsRequest['SubCategories'] = ["Hourly"]
        base_ObsRequest['Frequency'] = ["Hourly average"]
        base_ObsRequest['Parameters'] = dpe_selected_var_list 

        if self.Configuration.train_model:
            base_ObsRequest['StartDate'] = self.Configuration.train_start_date_aest.strftime('%Y-%m-%d')
            base_ObsRequest['EndDate'] = self.Configuration.train_end_date_aest.strftime('%Y-%m-%d')
        else:    
            base_ObsRequest['EndDate'] = self.Configuration.start_date_aest.strftime('%Y-%m-%d')
            base_ObsRequest['StartDate'] = self.Configuration.end_date_aest.strftime('%Y-%m-%d')

        #     print(ObsRequest['EndDate'])

        # asds

        ########### Set up use_file = False for downloading new OBS from API; If use_file = True will get the data from allobs.csv 
        # print(ObsRequest)
        # use_file = False
        # use_file = True
        # use_file = self.Configuration.use_file_training
        # print("---------------UseFileState-------------------")
        
        # print(self.use_file)

        ########################
        # Computing raw file names for in/out
        
        OBS_raw_save_file = self.OBS_save_file_template.format(
            raw_processed = "raw",
            input_stream = input_stream,
            region = self.Configuration.selected_region,
            var = self.Configuration.input_var_dir,
            additional_vars = self.Configuration.additional_var_dir, 
        )
        OBS_processed_save_file = self.OBS_save_file_template.format(
            raw_processed = "processed",
            input_stream = input_stream,
            region = self.Configuration.selected_region,
            var = self.Configuration.input_var_dir,
            additional_vars = self.Configuration.additional_var_dir, 
        )

        # Raw obs goes to the shared API input cache when configured.
        if self.Configuration.Main_output_run_shared_dir is not None:
            dir = self.Configuration.Main_output_run_shared_dir
        else:
            dir = self.Configuration.Main_output_run_full_dir

        processed_dir = (
            self.Configuration.Main_imputed_data_shared_dir
            or self.Configuration.Main_model_data_dir
        )
        legacy_inputs_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.Configuration.Main_model_data_dir)),
            "API_Input",
            "Inputs",
        )
        legacy_processed_candidates = sorted(
            glob.glob(
                os.path.join(
                    legacy_inputs_dir,
                    f"Allobs_processed_DPE_station_api_{self.Configuration.selected_region}_{self.Configuration.var_to_predict[0]}*.csv",
                )
            )
        )

        filename_full_path = os.path.join(dir, OBS_raw_save_file)
        processed_filename_full_path = os.path.join(processed_dir, OBS_processed_save_file)
        legacy_processed_path = None
        self.logger.info(
            "Checking legacy cache at {path} for region {region}".format(
                path=legacy_inputs_dir,
                region=self.Configuration.selected_region,
            ).ljust(self.justif - 2, '.') + 'OK'
        )

        def _load_last_timestamp(csv_path, target_tz):
            if not os.path.isfile(csv_path):
                return None

            cached_pd = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if isinstance(cached_pd.index, pd.DatetimeIndex) and len(cached_pd.index) > 0:
                cached_timestamp = cached_pd.index.max()
                if cached_timestamp.tzinfo is None:
                    return cached_timestamp.tz_localize(target_tz)
                return cached_timestamp.tz_convert(target_tz)

            if {'Date', 'Hour'}.issubset(cached_pd.columns):
                cached_dates = pd.to_datetime(cached_pd['Date'], errors='coerce')
                cached_hours = pd.to_numeric(cached_pd['Hour'], errors='coerce')
                cached_timestamps = cached_dates + pd.to_timedelta(cached_hours - 1, unit='h')
                cached_timestamp = cached_timestamps.max()
                if cached_timestamp is pd.NaT:
                    return None
                if cached_timestamp.tzinfo is None:
                    return cached_timestamp.tz_localize(target_tz)
                return cached_timestamp.tz_convert(target_tz)

            return None

        def _ensure_datetime_index(dataframe, target_tz='Australia/Brisbane'):
            dataframe = dataframe.copy()
            if not isinstance(dataframe.index, pd.DatetimeIndex):
                dataframe.index = pd.to_datetime(dataframe.index, errors='coerce')
            if dataframe.index.tz is None:
                dataframe.index = dataframe.index.tz_localize(target_tz)
            else:
                dataframe.index = dataframe.index.tz_convert(target_tz)
            dataframe.index.name = "datetime"
            return dataframe

        def _has_target_columns(dataframe, requested_target):
            normalized_target = "".join(ch for ch in str(requested_target).upper() if ch.isalnum())
            alias_groups = {
                "O3": ("O3", "OZONE"),
                "OZONE": ("O3", "OZONE"),
                "PM25": ("PM25", "PM2.5", "PM_25"),
                "PM2.5": ("PM25", "PM2.5", "PM_25"),
                "PM_25": ("PM25", "PM2.5", "PM_25"),
                "PM10": ("PM10",),
            }
            target_aliases = alias_groups.get(normalized_target, (requested_target,))
            prefixes = tuple(
                "{prefix}_".format(prefix=prefix)
                for prefix in target_aliases
            )
            return any(str(col).startswith(prefixes) for col in dataframe.columns)

        ########################
        AllObs_pd_existing = None
        processed_cache_pd = None
        existing_max_timestamp = None
        base_start_date = pd.Timestamp(base_ObsRequest['StartDate'])
        base_end_date = pd.Timestamp(base_ObsRequest['EndDate'])
        request_low_date = min(base_start_date, base_end_date)
        request_high_date = max(base_start_date, base_end_date)
        request_low_timestamp = request_low_date.tz_localize('Australia/Brisbane')
        request_high_timestamp = request_high_date.tz_localize('Australia/Brisbane')
        current_time = dtime.datetime.now(tz=request_low_timestamp.tzinfo)
        if not self.Configuration.train_model:
            request_high_timestamp = max(current_time, request_low_timestamp)

        if self.use_file:
            # Explicit use_file=True: skip download entirely, load from disk
            self.logger.info('Using OBS raw data file'.ljust(self.justif,'.'))
            AllObs_pd = pd.read_csv(filename_full_path, index_col=0)
            self.logger.info("Total rows loaded = {num}".format(num=len(AllObs_pd)))
            self.logger.info('OBS raw data = {msg}'.format(
                msg=OBS_raw_save_file).ljust(self.justif-6,'.') + 'LOADED')
            # Keep a persisted copy in the raw-only API input cache.
            self.MakeDir(dir)
            AllObs_pd.to_csv(filename_full_path)
            self.Change_permissions(filename_full_path)
            self.logger.info('OBS raw data = {msg}'.format(
                msg=OBS_raw_save_file).ljust(self.justif-6,'.') + 'WRITTEN')

        else:
            # Reuse cached raw data when possible and only download the missing tail.
            legacy_processed_bootstrap = False
            if legacy_processed_candidates:
                legacy_processed_path = legacy_processed_candidates[0]
                candidate_processed_pd = pd.read_csv(legacy_processed_path, index_col=0, parse_dates=True)
                if _has_target_columns(candidate_processed_pd, self.Configuration.var_to_predict[0]):
                    processed_cache_pd = candidate_processed_pd
                    existing_max_timestamp = _load_last_timestamp(legacy_processed_path, request_low_timestamp.tzinfo)
                    legacy_processed_bootstrap = True
                    self.logger.info(
                        f'Found legacy processed file for region "{self.Configuration.selected_region}" at '
                        f'{legacy_processed_path} with available time frame up to {existing_max_timestamp}'.ljust(
                            self.justif, '.') + 'REUSED')
                else:
                    processed_cache_pd = None
                    existing_max_timestamp = None
                    self.logger.info(
                        f'Ignoring legacy processed file for region "{self.Configuration.selected_region}" at '
                        f'{legacy_processed_path} because it does not contain target '
                        f'{self.Configuration.var_to_predict[0]}'.ljust(self.justif, '.') + 'SKIP')
                    legacy_processed_bootstrap = False
            elif os.path.isfile(processed_filename_full_path):
                processed_cache_pd = pd.read_csv(processed_filename_full_path, index_col=0, parse_dates=True)
                existing_max_timestamp = _load_last_timestamp(processed_filename_full_path, request_low_timestamp.tzinfo)
                self.logger.info(
                    f'Found processed file for region "{self.Configuration.selected_region}" '
                    f'with available time frame up to {existing_max_timestamp}'.ljust(self.justif, '.') + 'REUSED')
            else:
                processed_cache_pd = None
                existing_max_timestamp = None

            if os.path.isfile(filename_full_path):
                AllObs_pd_existing = pd.read_csv(filename_full_path, index_col=0)
                raw_cache_exists = True
                if existing_max_timestamp is None:
                    existing_max_timestamp = _load_last_timestamp(filename_full_path, request_low_timestamp.tzinfo)
                if existing_max_timestamp is not None:
                    self.logger.info(
                        f'Found raw file for region "{self.Configuration.selected_region}" '
                        f'with available time frame up to {existing_max_timestamp}'.ljust(self.justif, '.') + 'REUSED')
            else:
                AllObs_pd_existing = None
                raw_cache_exists = False

            if existing_max_timestamp is not None and existing_max_timestamp >= request_high_timestamp:
                self.logger.info(
                    f'File found for region "{self.Configuration.selected_region}" '
                    f'with available time frame up to {existing_max_timestamp}'.ljust(self.justif, '.') + 'REUSED')
                self.logger.info(
                    f'Requested download already covered up to {request_high_timestamp}'.ljust(self.justif, '.') + 'SKIPPED')
                if AllObs_pd_existing is not None:
                    AllObs_pd = AllObs_pd_existing
                elif processed_cache_pd is not None and _has_target_columns(processed_cache_pd, var_to_predict[0]):
                    AllObs_pd = None if legacy_processed_bootstrap else processed_cache_pd
                else:
                    AllObs_pd = pd.DataFrame()
            else:
                self.logger.info('Downloading DPE OBS data from API'.ljust(self.justif,'.'))
                if existing_max_timestamp is not None:
                    download_start_timestamp = max(existing_max_timestamp + pd.Timedelta(hours=1), request_low_timestamp)
                    self.logger.info(
                        f'Found file for region "{self.Configuration.selected_region}" with available time frame up to {existing_max_timestamp}. '
                        f'Downloading remaining data from {download_start_timestamp} to {request_high_timestamp}'.ljust(
                            self.justif, '.') + 'OK')
                else:
                    download_start_timestamp = request_low_timestamp
                    self.logger.info(
                        f'No existing file found for region "{self.Configuration.selected_region}". '
                        f'Downloading full requested range {request_low_timestamp} to {request_high_timestamp}'.ljust(
                            self.justif, '.') + 'OK')

                all_batch_data = []

                for batch_idx, station_batch in enumerate(station_batches, 1):
                    self.logger.info(f'Downloading batch {batch_idx}/{len(station_batches)} ({len(station_batch)} stations)'.ljust(self.justif,'.'))

                    ObsRequest = base_ObsRequest.copy()
                    ObsRequest['Sites'] = station_batch
                    ObsRequest['StartDate'] = download_start_timestamp.strftime('%Y-%m-%d')
                    ObsRequest['EndDate'] = request_high_timestamp.strftime('%Y-%m-%d')

                    max_attempts = 5
                    current_attempt = 1
                    batch_success = False

                    while current_attempt <= max_attempts:
                        AllObs = self.AQMS.get_historical_obs(ObsRequest)

                        if AllObs.status_code == 200:
                            batch_data = pd.json_normalize(AllObs.json())
                            all_batch_data.append(batch_data)
                            self.logger.info(f'  Batch {batch_idx} downloaded: {len(batch_data)} records'.ljust(self.justif-2,'.') + 'OK')
                            batch_success = True
                            break
                        else:
                            self.logger.error(f"  Batch {batch_idx} attempt {current_attempt} failed")
                            current_attempt += 1
                            if current_attempt <= max_attempts:
                                import time
                                time.sleep(2)

                    if not batch_success:
                        raise ValueError(f"Failed to download batch {batch_idx} after {max_attempts} attempts, error={AllObs.status_code}")

                # Combine batches into single new DataFrame
                if len(all_batch_data) > 1:
                    self.logger.info(f'Combining {len(all_batch_data)} batches'.ljust(self.justif,'.'))
                    AllObs_pd_new = pd.concat(all_batch_data, ignore_index=True)
                    self.logger.info(f'Combined data: {len(AllObs_pd_new)} total records'.ljust(self.justif-2,'.') + 'OK')
                else:
                    AllObs_pd_new = all_batch_data[0]

                # Merge with existing shared file if it exists
                self.MakeDir(dir)
                if AllObs_pd_existing is not None:
                    self.logger.info('Existing raw data found, merging with new download'.ljust(self.justif,'.'))
                    n_existing = len(AllObs_pd_existing)

                    AllObs_pd = pd.concat([AllObs_pd_existing, AllObs_pd_new], ignore_index=True)
                    duplicate_cols = ['Date', 'Hour', 'Site_Id', 'Parameter.ParameterCode']
                    if all(col in AllObs_pd.columns for col in duplicate_cols):
                        AllObs_pd = AllObs_pd.drop_duplicates(
                            subset=duplicate_cols,
                            keep='last'
                        ).reset_index(drop=True)

                    n_net_new = len(AllObs_pd) - n_existing
                    self.logger.info(f'Merged: {n_existing} existing + {len(AllObs_pd_new)} new = {len(AllObs_pd)} total ({n_net_new} net new rows)'.ljust(self.justif,'.') + 'OK')
                else:
                    self.logger.info('No existing raw data found, saving fresh download'.ljust(self.justif,'.'))
                    AllObs_pd = AllObs_pd_new

                # Save merged data back to shared file
                AllObs_pd.to_csv(filename_full_path)
                self.Change_permissions(filename_full_path)
                self.logger.info('OBS raw data = {msg}'.format(msg=OBS_raw_save_file).ljust(self.justif-7,'.') + 'WRITTEN')
                self.MakeDir(processed_dir)
                # print(AllObs_pd.tail())

            if AllObs_pd is None and processed_cache_pd is not None:
                # Legacy bootstrap path: the stored file is already in the
                # processed wide format expected by the downstream pipeline.
                Processed_obs_pd = _ensure_datetime_index(processed_cache_pd.sort_index())
                input_column_names = list(Processed_obs_pd.columns)
                self.dpe_selected_output_var_list = sorted(list(
                    itertools.chain.from_iterable(
                        [self.dpe_var_selection_dict[ii] for ii in self.Configuration.var_to_predict] )
                    ))

                output_column_names = sorted(["_".join([var, station]) for var in self.dpe_selected_output_var_list
                        for station in self.Configuration.dpie_output_station_list])

                print("-"*50)
                print ("input_column_names:")
                print (input_column_names)
                print("-"*30)
                print("output_column_names:")
                print(output_column_names)
                print("-"*50)

                self.MakeDir(processed_dir)
                Processed_obs_pd.to_csv(processed_filename_full_path)
                self.Change_permissions(processed_filename_full_path)
                self.logger.info('OBS processed data = {msg}'.format(
                    msg=OBS_processed_save_file).ljust(self.justif-6,'.') + 'WRITTEN')

                return Processed_obs_pd, input_column_names, output_column_names, dpe_species_properties_dict, dpe_selected_var_list
        

        ### specifies the desired data types for four columns:  Hour, Value, Parameter.ParameterCode, and Site_Id
        raw_required_columns = {"Hour", "Site_Id", "Parameter.ParameterCode", "Value"}
        if not raw_required_columns.issubset(set(AllObs_pd.columns)):
            if processed_cache_pd is not None and not processed_cache_pd.empty and _has_target_columns(processed_cache_pd, var_to_predict[0]):
                Processed_obs_pd = _ensure_datetime_index(processed_cache_pd.sort_index())
                input_column_names = list(Processed_obs_pd.columns)
                self.dpe_selected_output_var_list = sorted(list(
                    itertools.chain.from_iterable(
                        [self.dpe_var_selection_dict[ii] for ii in var_to_predict] )
                    ))

                output_column_names = sorted(["_".join([var, station]) for var in self.dpe_selected_output_var_list
                        for station in self.Configuration.dpie_output_station_list])

                print("-"*50)
                print ("input_column_names:")
                print (input_column_names)
                print("-"*30)
                print("output_column_names:")
                print(output_column_names)
                print("-"*50)

                self.MakeDir(processed_dir)
                Processed_obs_pd.to_csv(processed_filename_full_path)
                self.Change_permissions(processed_filename_full_path)
                self.logger.info('OBS processed data = {msg}'.format(
                    msg=OBS_processed_save_file).ljust(self.justif-6,'.') + 'WRITTEN')

                return Processed_obs_pd, input_column_names, output_column_names, dpe_species_properties_dict, dpe_selected_var_list

        dpe_Obs_dtype_dict = {
            "Hour"          : 'Int64',
            'Value'         : 'Float64',
            'Parameter.ParameterCode'  : 'string',
            'Site_Id'    : 'Int64',
        }
        AllObs_pd.astype(dpe_Obs_dtype_dict)
        # print("Columns of raw data:", AllObs_pd.columns)
      
        ######################
        # rearrange the pd to fit input format

        # print(AllObs_pd.loc[:,['DeterminingPollutant', 'Parameter.ParameterCode']])
        AllObs_pd["Date"] = pd.to_datetime(AllObs_pd['Date'], format='%Y-%m-%d')

        #remove 1 hour to set the values on the beginning of the hour
        AllObs_pd['Hour'] = pd.Series(
            [pd.Timedelta(hours=(ii-1)) for ii in AllObs_pd['Hour'].values]
            )
        # set index to dateime and localise it in AEST
        AllObs_pd.set_index(
            pd.DatetimeIndex((AllObs_pd["Date"] + AllObs_pd['Hour'])).tz_localize('Australia/Brisbane'), inplace=True
        )
        AllObs_pd.index.name = "datetime"
        
        ######################
        # Filter out future timestamps (beyond current time)
        # Get current time in the same timezone as data (AEST/Australia/Brisbane)
        current_time = dtime.datetime.now(tz=AllObs_pd.index.tzinfo)
        
        # Count records before filtering
        records_before = len(AllObs_pd)
        
        # Filter: keep only records with timestamp <= current time
        AllObs_pd = AllObs_pd[AllObs_pd.index <= current_time]
        
        # Count records after filtering
        records_after = len(AllObs_pd)
        records_removed = records_before - records_after
        
        if records_removed > 0:
            self.logger.info(f'Removed {records_removed} future timestamp(s)'.ljust(self.justif-2,'.') + 'OK')
            self.logger.info(f'Data filtered: {records_after} records remaining'.ljust(self.justif-2,'.') + 'OK')
        else:
            self.logger.info('No future timestamps found'.ljust(self.justif-2,'.') + 'OK')
        ######################
        
        columns_to_keep = ["Site_Id", "Parameter.ParameterCode", "Value"]
        Processed_obs_pd = AllObs_pd.loc[:,columns_to_keep]
        Processed_obs_pd["site_name"] = [self.dpe_reverse_station_site_detail_dict[ii] for ii in Processed_obs_pd["Site_Id"]]
        Processed_obs_pd["Var_name"] = Processed_obs_pd[["Parameter.ParameterCode","site_name"]].agg('_'.join, axis=1)
        columns_to_keep = ["Var_name", "Value"]
        Processed_obs_pd = Processed_obs_pd.loc[:,columns_to_keep]
        Processed_obs_pd = Processed_obs_pd.set_index(["Var_name",], drop=True, append=True).unstack(level="Var_name", fill_value=0)
        Processed_obs_pd.columns = [col[-1] for col in Processed_obs_pd.columns]
        Processed_obs_pd.sort_index(axis="columns", inplace=True)

        if processed_cache_pd is not None and not processed_cache_pd.empty:
            processed_cache_pd = _ensure_datetime_index(processed_cache_pd)
            Processed_obs_pd = _ensure_datetime_index(Processed_obs_pd)
            combined_processed_pd = pd.concat([processed_cache_pd, Processed_obs_pd], axis=0, sort=True)
            combined_processed_pd = combined_processed_pd[~combined_processed_pd.index.duplicated(keep='last')].sort_index()
            Processed_obs_pd = combined_processed_pd
            self.logger.info(
                'Merged legacy processed cache into current processed data'.ljust(self.justif - 2, '.') + 'OK'
            )
        # print(Processed_obs_pd.describe())
        
        ################# Fill NaN value at 1AM time missing ####################
        early_hours = Processed_obs_pd.index.hour < 2
        Fill = Processed_obs_pd.loc[early_hours].ffill()
        Processed_obs_pd.loc[early_hours] = Fill
        ################################################################
        # print(Processed_obs_pd)
        # print(Processed_obs_pd.describe())
        # ########################## Normalized#####################
                # from sklearn import preprocessing
                # cols = Processed_obs_pd.columns
                # ind = Processed_obs_pd.index
                # temp = Processed_obs_pd.values #returns a numpy array
                # min_max_scaler = preprocessing.MinMaxScaler()
                # x_scaled = min_max_scaler.fit_transform(temp)
                # Processed_obs_pd = pd.DataFrame(x_scaled, columns= cols, index=ind )

                # print(Processed_obs_pd.describe())
        
        ######################
        # name input and output columns
        # input 
        input_column_names = list(Processed_obs_pd.columns)
        # output
        self.dpe_selected_output_var_list = sorted(list(
            itertools.chain.from_iterable(
                [self.dpe_var_selection_dict[ii] for ii in self.Configuration.var_to_predict] )
                ))
        
        output_column_names = sorted(["_".join([var, station]) for var in self.dpe_selected_output_var_list 
                for station in self.Configuration.dpie_output_station_list])
        
        print("-"*50)
        print ("input_column_names:")
        print (input_column_names)
        print("-"*30)
        print("output_column_names:")
        print(output_column_names)
        print("-"*50)

        self.MakeDir(processed_dir)
        Processed_obs_pd.to_csv(processed_filename_full_path)
        self.Change_permissions(processed_filename_full_path)
        self.logger.info('OBS processed data = {msg}'.format(
            msg=OBS_processed_save_file).ljust(self.justif-6,'.') + 'WRITTEN')

        return Processed_obs_pd, input_column_names, output_column_names, dpe_species_properties_dict, dpe_selected_var_list

###########################################################################################
    def save_processed_data(self, input_data_pd, input_stream):
        """_summary_
        Save obs processed data 
        Returns:
            _type_: _description_
        """
        Output_Manager = OM.Output_manager_Class(self.Configuration)
        Output_Manager.output_obs_data(input_data_pd, input_stream)
        
        return        

###########################################################################################
    def filter_missing_values(self, input_data_pd, threshold_missing_data):
        """_summary_
        Check column with large missing values (>threshold_missing_data)
        Args:
            input_data_pd (_type_): _description_

        Returns:
            _type_: _description_
        """
        ### Check column with large missing values (>50%):
        mask =  (input_data_pd.isnull().sum() / len(input_data_pd)) > threshold_missing_data
        # columnx = input_data_pd_list[0].loc[:, mask]
        # print(input_data_pd_list[0])
        input_data_pd = input_data_pd.loc[:, ~mask]
        
        
        ###########################################################
        # print("*"*50)
        self.logger.info("Total % missing values for each column:")
        missing = 100*input_data_pd.isnull().sum() / len(input_data_pd)
        self.logger.info(missing)
        return input_data_pd

###########################################################################################
    def load_inputs(self,):
        """
        main function to run to trigger the input data flow
        """
        input_data_dict = {}
        
        for input_choice in self.Data_input_flow:

            if input_choice == "Hubert_files":
                var_data_list_from_input_pd = self.select_input_variables(self.additional_var_to_select)

                source_df, df_other, raw_CTM = self.load_Hubert_files()

                input_data_pd = pd.concat([df_other.loc[:,var_data_list_from_input_pd], source_df], axis=1)
                # print(input_data_pd)

                # Provide a consistent dict structure so callers can access
                # 'input_data', 'input_column_names', 'output_column_names', and 'specie_properties_dict'
                input_column_names = list(input_data_pd.columns)

                # Try to infer output columns for the configured target species (e.g. O3/OZONE, PM10, PM2.5)
                target = None
                try:
                    target = self.var_to_predict[0]
                except Exception:
                    target = None

                alias_groups = {
                    "O3": ("O3", "OZONE"),
                    "OZONE": ("O3", "OZONE"),
                    "PM25": ("PM25", "PM2.5", "PM_25"),
                    "PM2.5": ("PM25", "PM2.5", "PM_25"),
                    "PM_25": ("PM25", "PM2.5", "PM_25"),
                    "PM10": ("PM10",),
                }

                output_column_names = []
                if target is not None:
                    normalized_target = "".join(ch for ch in str(target).upper() if ch.isalnum())
                    aliases = alias_groups.get(normalized_target, (target,))
                    prefixes = tuple(f"{a}_" for a in aliases)
                    output_column_names = [c for c in input_column_names if str(c).startswith(prefixes)]

                specie_properties_dict = {}

                input_data_dict[input_choice] = {
                    "input_data": [input_data_pd],
                    "input_column_names": input_column_names,
                    "output_column_names": output_column_names,
                    "specie_properties_dict": specie_properties_dict,
                }
                self.logger.info("Hubert's data files loading".ljust(self.justif - 2 ,'.') + 'OK')

            elif input_choice == "Artificial_forecast":
                artificial_forecast_input_pd, input_column_names, output_column_names, artificial_species_properties_dict, var_data_list_from_input_pd = self.load_artificial_forecast_input()
                input_data_dict[input_choice] = {
                    "input_data" : [artificial_forecast_input_pd],
                    "input_column_names" : input_column_names,
                    "output_column_names" : output_column_names,
                    "specie_properties_dict": artificial_species_properties_dict
                }
                ### update Configure
                self.Configuration.var_data_list_from_input_pd = var_data_list_from_input_pd
                        
                ###############
                # saving data
                if self.Configuration.save_obs_processed_data:
                    self.save_processed_data(input_data_pd, input_choice)

                self.logger.info("Artificial_forecast data".ljust(self.justif - 2 ,'.') + 'OK')

            elif input_choice == "DPE_station_api":
                
                input_data_pd, input_column_names,  output_column_names, dpe_species_properties_dict, var_data_list_from_input_pd = self.load_dpe_api_input(input_choice)

                # filter missing values if training
                if self.Configuration.train_model:
                    input_data_pd = self.filter_missing_values(
                        input_data_pd, self.threshold_missing_data)
                
                input_data_dict[input_choice] = {
                    "input_data" : [input_data_pd],
                    "input_column_names" : input_column_names,
                    "output_column_names" : output_column_names,
                    "specie_properties_dict": dpe_species_properties_dict
                }
                ### update Configure
                self.Configuration.var_data_list_from_input_pd = var_data_list_from_input_pd

                ###############
                # saving data
                if self.Configuration.save_obs_processed_data:
                    self.save_processed_data(input_data_pd, input_choice)

                self.logger.info("DPE station api data".ljust(self.justif - 2 ,'.') + 'OK')

            else:
                self.logger.error("Input method".ljust(self.justif - 15 ,'.') + 'NOT IMPLEMENTED')
                sys.exit("Sorry :(")
            
        return input_data_dict
###########################################################################################
    def load_training_file(self, selected_region, iteration):
        """
        main function to load the training file
        """
        training_dir = self.Configuration.Main_Model_training_full_dir
        training_filename_template = self.Configuration.training_forward_output_filename_template
            
        training_filename = training_filename_template.format(
                region = self.selected_region, 
                mode = self.Configuration.train_forecast_dir.lower(),
                inputs = self.Configuration.n_steps_in, 
                outputs = self.Configuration.n_steps_out, 
                parameters = self.Configuration.var_to_predict[0], 
                iter=iteration,
                model = self.Configuration.forecast_method,
                )
        training_filename_fullpath = os.path.join(training_dir, training_filename)
        
        file_exist = os.path.isfile(training_filename_fullpath)
        training_pd = None
        if file_exist:
            training_pd = pd.read_csv(
                training_filename_fullpath, 
                infer_datetime_format=True, 
                parse_dates=["datetime"], 
                index_col=[ "datetime"])
            
        
        return training_pd, file_exist 

#################################################################################
    def load_evaluation_file_for_dashboard(self, iteration):
        """
        method to output the necessary for the produced forecast
        """ 
        forecast_method = self.Configuration.forecast_method
        dict_key_processed = forecast_method.rsplit("_")
        
        if 'BNN' in self.Configuration.forecast_method:
        # if 'BNN' in dict_key_processed:
            output_filename_template = self.Configuration.metrics_bnn_output_filename_template
            metric = "average_metric_BNN"
        else:
            output_filename_template = self.Configuration.metrics_output_filename_template
            metric = "average_metric"

        filename = output_filename_template.format(
            region = self.Configuration.selected_region,
            metrics = metric,
            var = self.Configuration.input_var_dir,
            inputs = self.Configuration.n_steps_in, 
            outputs = self.Configuration.n_steps_out, 
            additional_vars = self.Configuration.additional_var_dir, 
            iteration=iteration,
            model = self.Configuration.forecast_method,
            )
        dir = self.Configuration.Main_Model_training_evaluation_full_dir
        
        filename_full_path = os.path.join(dir, filename)
        print("eval file for dashboard")
        print(filename_full_path)
        file_exist = os.path.isfile(filename_full_path)
        evaluation_pd = None
        if file_exist:
            evaluation_pd = pd.read_csv(
                filename_full_path, 
                infer_datetime_format=True
                )
            self.logger.info('Evaluation metrics = {msg}'.format(msg=filename).ljust(self.justif-6,'.') + 'LOADED')
            
        return evaluation_pd, file_exist 

###########################################################################################
if __name__ == '__main__':
    import logging
    #from ..Tools import InitLogging as IL
    from ..Configuration import Configuration as CC
    justif = 102
################ logger init
    loggername='input_manager'
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
    fileh = logging.FileHandler(loggername+'.log')
    fileh.setLevel(logging.DEBUG)
# create console handler with a higher log level
    consoleh = logging.StreamHandler()
    consoleh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
    formatterfile    = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatterconsole = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    fileh.setFormatter(formatterfile)
    consoleh.setFormatter(formatterconsole)
# add the handlers to the logger
    logger.addHandler(fileh)
    logger.addHandler(consoleh)
################  
   
    start_date_utc = dtime.datetime(2017,1,1,0)
    # end_date_utc = start_date_utc + dtime.timedelta(days=2)
    end_date_utc = dtime.datetime(2020,1,1,0)
    # timestamp_aedt = dtime.datetime.now()
    timestamp_aedt = dtime.datetime.strptime('2023012014', '%Y%m%d%H')


###########################################################
    Configuration = CC.Configuration_Class(logger, justif,
            start_date_utc, end_date_utc, timestamp_aedt, 
            None, None, None, None, 
            None, None, None,

            )

    ##### input configuration
    Data_input_flow = ["Hubert_files",]
    # self.Data_input_flow = ["Artificial_forecast",]

    Configuration.configure_input(Data_input_flow)

    var_to_predict = ["O3"]
    additional_var_to_select = sorted(["Wind", "NO2"])

    dpie_input_station_list = ["BRINGELLY", "CAMPBELLTOWN_WEST", "CAMDEN", "LIVERPOOL"]
    lcs_input_station_list = []
    custom_input_station_list = []
    dpie_output_station_list = ["BRINGELLY", "CAMPBELLTOWN_WEST", "CAMDEN", "LIVERPOOL"]
    lcs_output_station_list = []
    custom_output_station_list = []
    
    n_steps_in = 12
    n_steps_out = 24

    Configuration.configure_model(None,
                        var_to_predict, additional_var_to_select,
                        dpie_input_station_list, lcs_input_station_list, custom_input_station_list,
                        dpie_output_station_list, lcs_output_station_list, custom_output_station_list,
                        n_steps_in, n_steps_out, None, None,
                        None, None,
                        )


    IMC = Input_manager_Class(Configuration)
    pollutant = "O3"
    # IMC.load_Hubert_files(pollutant)
    IMC.load_dpe_api_input()
    IMC.load_artificial_forecast_input()
