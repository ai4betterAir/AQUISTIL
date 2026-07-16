"""
..  module:: File_input
    :platform: Unix
    :synopsis: Definition of the basic object class to load files.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

import sys
import os
import stat
import pytz
import pandas as pd
# from ..Tools import DateMagics as DM

###########################################################################################
class File_input_Class(object):
    """ 
    This class defines a Input_Class, that contains the capacity to manage the inputs.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif,
                 ):

        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Configuring the File Input'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))



        return
###########################################################################################
    def load_prepared_obs_files(self, path, filename):
        """
        """
        file = os.path.join(path, filename)
        data_source_pd = pd.read_csv(file, header=0, infer_datetime_format=True,
                        dayfirst = True, parse_dates=['datetime'], index_col=['datetime'])

        aest = pytz.timezone('Australia/Brisbane')
        data_source_pd.index = pd.DatetimeIndex(data=data_source_pd.index, tz=aest,)

        data_source_pd = data_source_pd.resample('H').mean()
        null_percentage = data_source_pd.isnull().sum()/(0.01*len(data_source_pd))
        self.logger.info("loading file {ff}".format(ff= filename).ljust(self.justif-2,'.') + 'OK' )
        self.logger.info("Percentages of Null values = \n{pc}".format(pc=null_percentage))
        return data_source_pd


###########################################################################################
if __name__ == '__main__':
    import logging
    #from ..Tools import InitLogging as IL
    
    justif = 102
################ logger init
    loggername='Mapping'
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
 



