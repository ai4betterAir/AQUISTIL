"""
..  module:: region_station_mapping
    :platform: Unix
    :synopsis: Definition of the basic object class to spliot data.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
   
"""

###########################################################################################
class DPE_region_stations(object):
    """ 
    This class defines a Splitting_Class, that contains the capacity to manage the Splitting.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, var_to_predict):
        self.region_aliases = {
            "SYD": "Sydney",
            "ALLSYD": "Sydney",
            "SW": "SW_Sydney",
            "NW": "NW_Sydney",
            "CE": "CE_Sydney",
            "CW": "CW_Sydney",
            "LH": "Lower_Hunter",
            "UH": "Upper_Hunter",
        }

        targets = self._normalise_targets(var_to_predict)
        if len(targets) != 1:
            raise ValueError(
                "DPE_region_stations expects one target pollutant at a time. "
                "Expand multi-target runs before building the region map."
            )
        target = targets[0]

        ### Input list of stations
        ### For Ozone
        ### No Ozone for "VINEYARD", not much for "PROSPECT"
        ### CE Sydney: No Ozone for "ALEXANDRIA", "CHULLORA", "LINDFIELD"
        ### Newcastle: No Ozone for "CARRINGTON"
        # Select_Sydney = ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE",  "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK",  "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS" ]  ### for PM2.5
        
        ozone_regions = {
            "CC_Coast": ["WYONG"],
            "CE_Sydney": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CN_Table": ["BATHURST", "ORANGE"],
            "Lower_Hunter": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "NW_Sydney": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "Newcastle": ["STOCKTON"],
            "SR_Table": ["GOULBURN"],
            "SW_Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "Upper_Hunter": ["MERRIWA", "MUSWELLBROOK", "SINGLETON"]
        }
        pm10_regions = {
            "CC_Coast": ["WYONG"],
            "CE_Sydney": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CN_Table": ["BATHURST", "ORANGE"],
            "Lower_Hunter": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "NR_Table": ["ARMIDALE"],
            "NW_Sydney": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "Newcastle": ["CARRINGTON", "MAYFIELD", "STOCKTON"],
            "SR_Table": ["GOULBURN"],
            "SW_Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "Upper_Hunter": ["ABERDEEN", "BULGA", "CAMBERWELL", "JERRYS_PLAINS", "MAISON_DIEU", "MERRIWA", "MOUNT_THORLEY", "MUSWELLBROOK", "MUSWELLBROOK_NW", "SINGLETON", "SINGLETON_NW", "SINGLETON_SOUTH", "WARKWORTH", "WYBONG"]
        }
        pm25_regions = {
            "CC_Coast": ["WYONG"],
            "CE_Sydney": ["ALEXANDRIA", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "MACQUARIE_PARK", "RANDWICK", "ROZELLE", "ULTIMO_UTS"],
            "CN_Table": ["BATHURST", "ORANGE"],
            "Lower_Hunter": ["BERESFIELD", "NEWCASTLE", "WALLSEND"],
            "NR_Table": ["ARMIDALE"],
            "NW_Sydney": ["PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RICHMOND", "ROUSE_HILL", "ST_MARYS"],
            "Newcastle": ["CARRINGTON", "MAYFIELD", "STOCKTON"],
            "SR_Table": ["GOULBURN"],
            "SW_Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "LIVERPOOL", "OAKDALE"],
            "Sydney": ["BRINGELLY", "CAMDEN", "CAMPBELLTOWN_WEST", "COOK_AND_PHILLIP", "EARLWOOD", "LIDCOMBE", "LIVERPOOL", "MACQUARIE_PARK", "OAKDALE", "PARRAMATTA_NORTH", "PENRITH", "PROSPECT", "RANDWICK", "RICHMOND", "ROUSE_HILL", "ROZELLE", "ST_MARYS"],
            "Upper_Hunter": ["CAMBERWELL", "MERRIWA", "MUSWELLBROOK", "SINGLETON"]
        }

        self.region_stations_by_target = {
            "O3": ozone_regions,
            "PM2.5": pm25_regions,
            "PM10": pm10_regions,
        }
        self.available_targets = sorted(self.region_stations_by_target)

        # self.DPE_region_stations_dict = {
        #     "SW_Sydney" : SW_Sydney,
        #     "NW_Sydney" : NW_Sydney,
        #     "CE_Sydney" : CE_Sydney,
        #     "CW_Sydney" : CW_Sydney,
        #     "Illawara" : Illawara,
        #     "Lower_Hunter" : Lower_Hunter,
        #     "Upper_Hunter" : Upper_Hunter,
        #     "Newcastle" : Newcastle,
        # }
        # self.DPE_region_stations_dict = {
        #     #"ALLSYD" : ALL_Sydney,
        #     "SLSYD" : Select_Sydney,
        # }

        if target not in self.region_stations_by_target:
            raise ValueError(
                "No DPE region station map is configured for target "
                "{target}. Available targets: {available}".format(
                    target=target,
                    available=", ".join(self.available_targets),
                )
            )

        print("==== Stations for {target} ====".format(target=target))
        self.target = target
        self.DPE_region_stations_dict = self.region_stations_by_target[target]
        if isinstance(self.DPE_region_stations_dict, tuple):
            self.DPE_region_stations_dict = self.DPE_region_stations_dict[0]

        return

    def _normalise_targets(self, var_to_predict):
        if isinstance(var_to_predict, str):
            return [var_to_predict]

        targets = []
        for value in var_to_predict:
            if isinstance(value, (list, tuple)):
                targets.extend(self._normalise_targets(value))
            else:
                targets.append(value)
        return targets

    def resolve_region_name(self, selected_region):
        return self.region_aliases.get(selected_region, selected_region)
