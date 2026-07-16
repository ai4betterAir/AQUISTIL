"""
.. module:: api_client
   :platform: Unix
   :synopsis: Everything needed to use the api to query aqms.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>


"""
import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
import urllib
import datetime as dt
import json

###########################################################################################
class NSWAirQualityAPIClient(object):
    """
    This class defines and configures the api to query the aqms database
    """
    def __init__(self, ): 

        self.logger = logging.getLogger(__file__)
        # self.url_api = "https://dpe-im-api-airquality-uat.azurewebsites.net"
        self.url_api = "https://data.airquality.nsw.gov.au"
        self.headers = {'content-type': 'application/json', 'accept': 'application/json'}

        # Default HTTP timeout (seconds). Override via env var if needed.
        try:
            self.timeout = int(os.environ.get('AQMS_API_TIMEOUT', '180'))
        except Exception:
            self.timeout = 180

        # Retry only failures that are normally transient. POST retries are safe
        # here because the API endpoints are read-only observation queries.
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(("GET", "POST")),
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        
        self.get_site_url = 'api/Data/get_SiteDetails'
        self.get_parameters = 'api/Data/get_ParameterDetails'
        self.get_observations = 'api/Data/get_Observations'  
        return

    def _request(self, method, url, **kwargs):
        """Issue and validate an API request, preserving useful diagnostics."""
        kwargs.setdefault("headers", self.headers)
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.request(method, url, **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            preview = (response.text or "").strip().replace("\n", " ")[:300]
            raise RuntimeError(
                "NSW Air Quality API returned HTTP %s for %s; response=%r"
                % (response.status_code, response.url, preview)
            ) from exc

        body = response.text or ""
        if not body.strip():
            raise RuntimeError(
                "NSW Air Quality API returned an empty response (HTTP %s) for %s"
                % (response.status_code, response.url)
            )
        try:
            response.json()
        except ValueError as exc:
            preview = body.strip().replace("\n", " ")[:300]
            content_type = response.headers.get("Content-Type", "unknown")
            raise RuntimeError(
                "NSW Air Quality API returned non-JSON content (HTTP %s, "
                "Content-Type %s) for %s; response=%r"
                % (response.status_code, content_type, response.url, preview)
            ) from exc
        return response

###########################################################################################
    def get_site_details(self, ):
        '''
        Build a query to return all the sites details
        '''
        query = urllib.parse.urljoin(self.url_api, self.get_site_url)
        # print(query)
        # response = requests.post(url = query, data = '')
        response = self._request("GET", query)
        return response

###########################################################################################
    def get_parameters_details(self, ):
        '''
        Build a query to return all the sites details
        '''
        query = urllib.parse.urljoin(self.url_api, self.get_parameters)
        #print(query)
        # response = requests.post(url = query, data = '')
        response = self._request("GET", query)
        return response

###########################################################################################
    def get_historical_obs(self, ObsRequest):
        '''
        Build a query to return all the sites details
        '''
        query = urllib.parse.urljoin(self.url_api, self.get_observations)
        
        response = self._request(
            "POST",
            query,
            data=json.dumps(ObsRequest),
        )
        return response

###########################################################################################
    def get_now_obs(self, ):
        '''
        Build a query to return all the sites details
        '''
        query = urllib.parse.urljoin(self.url_api, self.get_observations)
        
        response = self._request("POST", query, data='')
        return response

###########################################################################################
    def ObsRequest_init(self, ):
        '''
        Build a empty dictionary to ready to post to get the obs
        '''
        ObsRequest = {}
        ObsRequest['Parameters'] = []
        ObsRequest['Sites'] = []
        ObsRequest['StartDate'] = ''
        ObsRequest['EndDate'] = ''
        ObsRequest['Categories'] = []
        ObsRequest['SubCategories'] = []
        ObsRequest['Frequency'] = []
        
        return ObsRequest

###########################################################################################

if __name__ == '__main__':    
    import pandas as pd    
    
    AQMS = NSWAirQualityAPIClient()
    
    ObsRequest = AQMS.ObsRequest_init()
    StartDate = dt.datetime(2020,1,1,12)
    EndDate = dt.datetime(2020,1,2,12)
    
    AllSites = AQMS.get_site_details()
    print(pd.json_normalize(AllSites.json()))

    for i, site in enumerate(AllSites.json()):
        #print(i, site)
        ObsRequest['Sites'].append(site['Site_Id'])
    
    Allparameters = AQMS.get_parameters_details()
    print(pd.json_normalize(Allparameters.json()))
    for i, param in enumerate(Allparameters.json()):
        #print(i, param)
        ObsRequest['Parameters'].append(param['ParameterCode'])
        ObsRequest['Categories'].append(param['Category'])
        ObsRequest['SubCategories'].append(param['SubCategory'])
        ObsRequest['Frequency'].append(param['Frequency'])
    
    #make all list unique
    ObsRequest['Parameters'] = list(set(ObsRequest['Parameters']))
    ObsRequest['Categories'] = list(set(ObsRequest['Categories']))
    ObsRequest['SubCategories'] = list(set(ObsRequest['SubCategories']))
    ObsRequest['Frequency'] = list(set(ObsRequest['Frequency']))
    
    ObsRequest['StartDate'] = StartDate.strftime('%Y-%m-%d')
    ObsRequest['EndDate'] = EndDate.strftime('%Y-%m-%d')

    ObsRequest_now = AQMS.ObsRequest_init()
    AllObs = AQMS.get_historical_obs(ObsRequest)
    # AllObs = AQMS.get_Obs(ObsRequest_now)
    #ObsRequest['Sites'] = [190]
    #ObsRequest['Parameters'] =  ['WDR']
    #ObsRequest['Categories'] =  ['Averages']
    #ObsRequest['SubCategories'] = ['Hourly']
    #ObsRequest['Frequency'] =  ['Hourly average']
    
    #AllObs = AQMS.get_Obs(ObsRequest)
    # print(json.dumps(ObsRequest))
    df = pd.json_normalize(AllObs.json(), )
    # print(AllObs.json())
    print(df.columns)
    print(df)
    print(pd.json_normalize(AQMS.get_now_obs().json()))
    # for i, obs in enumerate(AllObs.json()):
    #     print(i, obs.text)
        
