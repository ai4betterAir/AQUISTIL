"""
..  module:: Forecast_Stats
    :platform: Unix
    :synopsis: Grab and concatenate the forecast files for further evaluation.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>

"""

import sys
import os
import stat
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Core_iHPC.Tools import InitLogging as IL
from Core_iHPC.Configuration import Configuration as CC
from Core_iHPC.Inputs import Input_manager as IM
from Core_iHPC.Tools import DateMagics as DM
from Core_iHPC.Evaluation import Evaluation_singlemodel as EVSM
import itertools as it
import pathlib
import fnmatch

###########################################################################################

class Forecast_Stats(object):
    """ 
    This class defines a Input_Class, that contains the capacity to manage the intputs.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif,
                # yaml_config_filename,
                ):

        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Forecast_Stats'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        self.config_from_file = None
        
        
        # configuration files
        # self.yaml_config_filename = yaml_config_filename
        
        # self.input_data_dir = "data"
        
        # self.Main_training_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Training"
        # self.Main_output_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Forecast"
        # self.Main_model_data_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Model_weights"
        # self.Code_configuration_main_dir = "/home/barthelemyx/Projects/Deep_learning/cnn_lstm_forecast/Core_iHPC/Tools/Config_testing"
        self.nownownow = dt.datetime.now().strftime("%Y%m%d")


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
    def List_forecast_files(self,
            Main_training_dir,
            Main_output_dir,
            Main_model_data_dir,
            Code_configuration_main_dir,
            ):
        """_summary_
        Forecast directory arborescence:
        
        self.Main_output_dir   
            / self.Region_dir 
                / sel.input_var_dir 
                    / self.additional_var_dir 
                        / self.Model_dir 
                            / self.run_output_dir (name convention) 
                                / files
                        / Model_n  ...                                                                                                                         |-> self.Output_run_status_dir (json)

        """
        
        # exclude dashboard files
        exclude = set(["Dashboard"])
        forecast_files = []
        
        
        for root, dirs, files in os.walk(pathlib.Path(Main_output_dir), topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            for ff in files:
                if fnmatch.fnmatch(ff, "*_forecast_*.csv"):
                #     print(
                #     "root = ", root, 
                #     "dirs = ", dirs,
                #     "files = ", ff
                # )
                    forecast_files.append(os.path.join(root, ff))
        print(forecast_files)
        return forecast_files   
###########################################################################################
    def process_individual_forecast_file(self, file_full_path):
        
        # finding forecast parameters
        # split by path
        splitted = pathlib.Path(file_full_path).parts
        # print(splitted)
        filename = splitted[-1]
        run_output_dir = splitted[-2]
        Model_dir = splitted[-3]
        additional_var_dir = splitted[-4]
        input_var_dir = splitted[-5]
        Region_dir = splitted[-6]
        
        # split filname for options
        
        splitted = pathlib.Path(filename).stem.split("_")
        date = splitted[-1]
        n_retrain = splitted[-2]
        len_output = splitted[-3]
        len_input = splitted[-4]
        
        # split name convention dir
        splitted = run_output_dir.split("---tstamp---")
        forecast_date = splitted[0]
        timestamp = splitted[-1]
         

        # print(
        #     "region=", Region_dir,"\n",
        #     "input_var=", input_var_dir,"\n",
        #     "additional_var=", additional_var_dir,"\n",
        #     "Model=", Model_dir,"\n",
        #     "run_output_dir=",run_output_dir,"\n",
        #     "filename=",filename,"\n",
        #     "len_input=", len_input, "\n",
        #     "len_output=",len_output, "\n",
        #     "n_retrain=",n_retrain,"\n",
        #     "date=", date,"\n",
        #     "forecast_date=", forecast_date,"\n",
        #     "timestamp=",timestamp,"\n",
        # )

        # read file with pandas
        forecast_pd = pd.read_csv(file_full_path)

        forecast_pd["region"] = Region_dir
        forecast_pd["input_var"] = input_var_dir
        forecast_pd["additional_var"] = additional_var_dir
        forecast_pd["Model"] = Model_dir
        forecast_pd["run_output_dir"] = run_output_dir
        forecast_pd["filename"] = filename
        forecast_pd["len_input"] = len_input
        forecast_pd["len_output"] = len_output
        forecast_pd["n_retrain"] = n_retrain
        # forecast_pd["date"] = date
        forecast_pd["forecast_date"] = forecast_date
        forecast_pd["timestamp"] = timestamp
        
        # print(forecast_pd)
        
        return forecast_pd
###########################################################################################
    def process_all_forecast_files(self,
            Main_training_dir,
            Main_output_dir,
            Main_model_data_dir,
            Code_configuration_main_dir,
            ):

        forecast_files_list = self.List_forecast_files(
            Main_training_dir,
            Main_output_dir,
            Main_model_data_dir,
            Code_configuration_main_dir,
            )
        all_forecast_pd_list = []
        for ff in forecast_files_list:
            # print(ff)
            individual_forecast_pd = self.process_individual_forecast_file(ff)
            all_forecast_pd_list.append(individual_forecast_pd)
        

        all_forecast_pd = pd.concat(all_forecast_pd_list, 
                                    axis=0, 
                                    ignore_index=True,
        
                                    )
        # convert datime column                            
        all_forecast_pd['datetime'] = pd.to_datetime(all_forecast_pd['datetime'])
        
        dtype_dict = {
            "forecast_hours" : "Int64"
        }
        
        all_forecast_pd = all_forecast_pd.astype(dtype_dict)
        
        # set the index
        Index_levels = [
            "input_var",
            "additional_var",
            "Model",
            "region",  
            "forecast_date",
            "len_input",
            "len_output",
            "n_retrain",
            'datetime', 
            'forecast_hours',
            'run_output_dir',
            'filename',
            'timestamp',
            ]        
        
        
        
        all_forecast_pd = all_forecast_pd.set_index(
                                        Index_levels,
                                        drop=True,
                                        append=False,
                                        
                                    )
                                    
        var_list, station_list = self.var_and_station_list_from_columns(list(all_forecast_pd.columns))
        # print(all_forecast_pd)
        return all_forecast_pd, var_list, station_list
        
###########################################################################################
    def var_and_station_list_from_columns(self,
                                          column_names,
                                          ):
        var_list = []
        station_list = []
        for col in column_names:
            # split only once for var and station
            splitted = col.split("_", maxsplit=1)
            var_list.append(splitted[0])
            station_list.append(splitted[-1])
        # make names unique
        var_list = list(set(var_list))
        station_list = list(set(station_list))
        var_list = ["O3" if var=="OZONE" else var for var in  var_list]
        # print(var_list)
        # print(station_list)
        return var_list, station_list
    
###########################################################################################
    def get_obs_from_api(self,
                         all_datetime,
                         var_list_from_forecast,
                         station_list_from_forecast,
                         Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir,
                         n_days_to_substact,
                         ):
        
        #### Start date of data for forecasting
        # self.train_start_date_utc = dt.datetime.strptime(all_datetime[0])
        self.train_start_date_utc, self.train_start_date_aest, self.train_start_date_aedt = DM.DateMagics(all_datetime[0], "AEST")
        #### End date of getting data for forecasting
        # self.train_end_date_utc = dt.datetime.strptime(all_datetime[-1])
        self.train_end_date_utc, self.train_end_date_aest, self.train_end_date_aedt = DM.DateMagics(all_datetime[-1], "AEST")

        self.start_date_utc = self.train_start_date_utc
        self.end_date_utc = self.train_end_date_utc
        self.timestamp_utc, self.timestamp_aest, self.timestamp_aedt = DM.DateMagics(dt.datetime.now(), "AEDT")
        
        # api only serve data older than a few days
        # n_days_to_substact = 90
        self.end_date_utc = min(self.end_date_utc, self.timestamp_utc - dt.timedelta(days=n_days_to_substact))
        self.train_end_date_utc = min(self.train_end_date_utc, self.timestamp_utc - dt.timedelta(days=n_days_to_substact))
             


        self.Forecast_RunTimeHours = 1
        self.Forecast_NumberofCpu = 1
        self.Forecast_Partition = "toto"


    ###########################################################
        self.Configuration = CC.Configuration_Class(self.logger, self.justif,
            self.train_start_date_utc, self.train_end_date_utc, 
            self.start_date_utc, self.end_date_utc, self.timestamp_utc, 
            self.input_data_dir,
            Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir,  
            # self.base_output_dir, self.model_forecast_save_dir, self.base_run_dir, 
            self.Forecast_RunTimeHours, self.Forecast_NumberofCpu, self.Forecast_Partition, 
#            self.selected_region, 
#            self.var_to_predict,  self.full_input_pd,

            )
        
        self.forecast_method = None
        
        self.var_to_predict = var_list_from_forecast
        self.additional_var_to_select = []
            
        self.DPE_region_stations_dict = {}
        self.selected_region  = "mixed"

        self.n_steps_in = None
        self.n_steps_out = None
        self.n_epochs = None
        self.batch_size = None

        self.dpie_input_station_list = station_list_from_forecast
        self.lcs_input_station_list = []
        self.custom_input_station_list = []

        ### Output list of stations
        self.dpie_output_station_list = []
        self.lcs_output_station_list = []
        self.custom_output_station_list = []

        #### Defining the training model -> True, of skipping training and making the forecast with pretrained model --> False
        self.train_model = True
        # self.train_model = False
        self.save_model = False
        self.model_parameters_dict = {}

        self.Configuration.Main_Model_training_full_dir = Main_training_dir
        self.Configuration.Main_output_run_full_dir = Main_training_dir        
        self.Configuration.configure_model(self.forecast_method,
                            self.var_to_predict, self.additional_var_to_select,
                            self.DPE_region_stations_dict, self.selected_region,  
                            self.dpie_input_station_list, self.lcs_input_station_list, self.custom_input_station_list,
                            self.dpie_output_station_list, self.lcs_output_station_list, self.custom_output_station_list,
                            self.n_steps_in, self.n_steps_out, self.n_epochs, self.batch_size,
                            self.train_model, self.save_model, #self.selected_region, self.full_input_pd
                            self.model_parameters_dict,
                            )
        ### Using data collected from API
        self.Data_input_flow = ["DPE_station_api"]
        self.use_file = False
        # self.use_file = False
        # self.use_file = True
        
        self.save_obs_processed_data = False
        # self.save_obs_processed_data = True
        
        self.Configuration.configure_input(self.Data_input_flow,  
                                            self.use_file, 
                                            self.save_obs_processed_data)

        self.IMC = IM.Input_manager_Class(self.Configuration)
        # self.IMC.load_dpe_api_input("DPE_station_api")
        input_data_dict = self.IMC.load_inputs()
        # print(input_data_dict)
        
        return input_data_dict
###########################################################################################
    def grouper(self, n, iterable, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return it.zip_longest(fillvalue=fillvalue, *args )       


###########################################################################################
    def get_all_obs(self,
                         all_datetime,
                         var_list_from_forecast,
                         station_list_from_forecast,
                         Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir,
                         n_days_to_substact,
                         ):

        # need to iterate on station because API is crap
        obs_from_api_list = []
        group_station_by = 4
        for station_tuple in self.grouper(group_station_by, station_list_from_forecast):
            
            station_list = [st for st in station_tuple if st is not None]
            obs_data_dict = self.get_obs_from_api(
                            all_datetime,
                            var_list_from_forecast,
                            station_list,
                            Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir,
                            n_days_to_substact,
                            )
            obs_from_api_list.append(obs_data_dict['DPE_station_api']["input_data"][0])
        obs_from_api_pd = pd.concat(obs_from_api_list, axis=1)
        return obs_from_api_pd

###########################################################################################
    def reorganise_table_format(self,
            All_data_pd,
            var_list_from_forecast, station_list_from_forecast
        ):

        groupby_levels = [
            "input_var",
            ]
        
        grouped = All_data_pd.groupby( 
                                      by = groupby_levels,
                                      as_index=True,
                                      dropna = False,
                                      )
        forecast_reshaped_pd_list = []
        for key, group in grouped:
            # print(key,)
            # print( group)
            input_var = key[0]
            # select columns
            all_columns_names = list(group.columns)
            selected_var_columns = [col for col in all_columns_names if col.startswith(input_var)]
            # print( selected_var_columns)
            # print(group[selected_var_columns])
            # keep forecast columns only from region i.e. columns without NAN
            # column_forecast_list = [
            #    col for col in  selected_var_columns
            #    if group[col].notna().all()
            # ]
            # print(column_forecast_list)
            
            column_forecast_list =selected_var_columns
            
            
            for col in column_forecast_list:
                forecast_pd = group.loc[:,[col]]
                # print("*"*80)
                # print(group)
                obs_col_name = "target_{col}".format(col= col)
                obs_pd = group.loc[:,[obs_col_name]]
                
                station_name = col.split("_",1)[-1]
                
                rename_col_dict = {col:station_name}
                forecast_reshaped_pd = forecast_pd.rename(columns = rename_col_dict, inplace = False)

                # print("-"*100)
                # print("rename dict", rename_col_dict)
 
                rename_col_dict = {obs_col_name:station_name}
                obs_reshaped_pd = obs_pd.rename(columns = rename_col_dict, inplace = False)

                # print("rename dict2", rename_col_dict)
                
                forecast_reshaped_pd = forecast_reshaped_pd.stack().rename_axis(index={None:"station_name"}).to_frame(name = "forecast")
                obs_reshaped_pd = obs_reshaped_pd.stack().rename_axis(index={None:"station_name"}).to_frame(name = "obs")
                
                final_pd = pd.concat([forecast_reshaped_pd, obs_reshaped_pd], axis = 1)
                
                forecast_reshaped_pd_list.append(final_pd)

                # print(final_pd.index, final_pd.columns)
                # print("|"*100)
                # stop()
        
        All_data_reshaped_pd = pd.concat(forecast_reshaped_pd_list, axis = 0)     
        
        return All_data_reshaped_pd   

###########################################################################################
    def compute_stats(self,
                      All_data_pd,
                      groupby_levels):

        
        # remove rows with nans
        All_data_pd = All_data_pd.dropna()
        
        
        grouped = All_data_pd.groupby( 
                                      by = groupby_levels,
                                      as_index=True,
                                      dropna = False,
                                      )
        Stats_pd_list = []
        for key, group in grouped:
            # print("*"*80)
            # print(key,)
            EC = EVSM.Evaluation_class(group["forecast"], group["obs"])
            
            data_dict = {
                "mae" : EC.mae(),
                "mape" : EC.mape(),
                "rmse" : EC.rmse(),
                "pearson_r" : EC.pearson_r(),
                "r_square" : EC.r_square(),
                "index_of_agreement" : EC.index_of_agreement(),
                "max_bias_error" : EC.max_bias_error()
            }
            # print(data_dict)
            # print(key, groupby_levels)
            
            multiindex = pd.MultiIndex.from_tuples([key], names=groupby_levels)
            
            Stats_pd = pd.DataFrame(
                data = data_dict, index = multiindex
            )
            
            Stats_pd_list.append(Stats_pd)
            # print( Stats_pd)
            # input_var = key[0]
            # select columns
            
            # build df for eval
            # print("|"*80)
        
        Stats_pd = pd.concat(Stats_pd_list) 
        return Stats_pd
###########################################################################################
    def run_all(self,):
        """
        """


        self.input_data_dir = "data"
        
        # Main_training_dir = "./Eval_data"
        # Main_output_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Forecast"
        # Main_model_data_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Model_weights"
        # Code_configuration_main_dir = "/home/barthelemyx/Projects/Deep_learning/cnn_lstm_forecast/Core_iHPC/Tools/Config_testing"
        #########
        # New place
        Main_training_dir = "/mnt/scratch_lustre/ar_aichem_scratch/AI_Runs/Training"
        Main_output_dir = "/mnt/scratch_lustre/ar_aichem_scratch/AI_Runs/Forecast"
        Main_model_data_dir = "/mnt/scratch_lustre/ar_aichem_scratch/AI_Runs/Model_weights"
        Code_configuration_main_dir = "/mnt/scratch_lustre/ar_aichem_scratch/Scripts/cnn_lstm_forecast/Core_iHPC/Tools/Config_testing"


    ###########################################################
    # retrieve all forecast from files
        all_forecast_pd, var_list_from_forecast, station_list_from_forecast = self.process_all_forecast_files(
            Main_training_dir,
            Main_output_dir,
            Main_model_data_dir,
            Code_configuration_main_dir,
            )
        # print(station_list_from_forecast)
        # print(all_forecast_pd.index)
        # print(all_forecast_pd.columns)
        # print(all_forecast_pd)

    ###########################################################
    # build the api request to get the obs
        
        all_datetime = all_forecast_pd.index.get_level_values("datetime").sort_values()
        print("datetime bounds from forecast files = ", all_datetime[0], all_datetime[-1])

        # api only serve data older than a few days
        n_days_to_substact = 3
        
        use_api = True
        # use_api = False
        
        obs_filename = "./Obs_data.csv"
        if use_api:
            # call the api and save data
            obs_from_api_pd = self.get_all_obs(
                            all_datetime,
                            var_list_from_forecast,
                            station_list_from_forecast,
                            Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir,
                            n_days_to_substact,
                            )
            # print(obs_from_api_pd)
            
            # rename columns
            rename_dict = {
                col: "target_{col}".format(col = col) 
                for col in obs_from_api_pd.columns
            }
            obs_from_api_pd = obs_from_api_pd.rename(columns=rename_dict)
            # save the file
            obs_from_api_pd.to_csv(obs_filename)
        else:
            #use presaved data
            obs_from_api_pd = pd.read_csv(obs_filename)    
            obs_from_api_pd['datetime'] = pd.to_datetime(obs_from_api_pd['datetime'])
            obs_from_api_pd.set_index(
                  ['datetime'],
                  drop=True,
                  inplace=True  
                )
        # print(obs_from_api_pd)
    ###########################################################
    # merge forecast and obs on datetime
        forecast_index = list(all_forecast_pd.index.names)

        All_data_pd = all_forecast_pd.join(
                obs_from_api_pd, 
                how = 'left', 
                on=["datetime"], 
                # left_index = True,
                # right_on = ["datetime"],
                sort=False, 
                )

        
        
        # print(All_data_pd)  
        
        # keep only forecasts
        # mask = All_data_pd["forecast_hours"] > 0
        
        All_data_pd = All_data_pd.query('forecast_hours > 0')
        # print(All_data_pd)  
        
        All_data_pd = self.reorganise_table_format(
            All_data_pd,
            var_list_from_forecast, station_list_from_forecast,
        )
        
        # print(All_data_pd)  
        
        # stop()
        
        All_data_filename_template = "All_data_for_stats_eval_{datetime}.csv"
        Stats_filename_template = "Statistical_eval_{datetime}.csv"
        
        All_data_pd.to_csv(All_data_filename_template.format(datetime=self.nownownow))
        # All_data_pd = pd.read_csv("./Join_result.csv")
        
        groupby_levels = [
            "input_var",
            # "additional_var",
            "Model",
            "region",  
            # "forecast_date",
            "len_input",
            # "len_output",
            # "n_retrain",
            # 'datetime', 
            # 'forecast_hours',
            # 'run_output_dir',
            # 'filename',
            # 'timestamp',
            "station_name",
            ]
        
        
        Stats_pd = self.compute_stats(
                      All_data_pd,
                      groupby_levels)

        
        print( Stats_pd)
        Stats_pd.to_csv(Stats_filename_template.format(datetime=self.nownownow))
        return
###########################################################################################
if __name__ == '__main__':
    
    ################ logger init #################
    justif = 102
    loggername= "Forecast_Stats"
    logger = IL.Initialise_logging(loggername)
    
    FCC = Forecast_Stats(logger, justif,
                )
    
    forecast_files_list = FCC.run_all()

   
