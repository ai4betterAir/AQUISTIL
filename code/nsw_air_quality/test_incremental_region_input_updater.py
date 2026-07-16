import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from nsw_air_quality import incremental_region_input_updater as updater


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeAQMS:
    responses = []
    requests = []

    def ObsRequest_init(self):
        return {}

    def get_historical_obs(self, request):
        self.__class__.requests.append(request)
        return self.__class__.responses.pop(0)


def _observations_for_day():
    return [
        {
            "Date": "2026-07-10",
            "Hour": hour + 1,
            "Site_Id": 1,
            "Value": 100 + hour,
            "Parameter": {"ParameterCode": "PM2.5"},
        }
        for hour in range(24)
    ]


class IncrementalWideInputTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(
            self.tempdir.name,
            "Allobs_processed_DPE_station_api_Test_ALL.csv",
        )
        times = pd.date_range("2026-07-10 00:00", "2026-07-10 10:00", freq="h", tz=updater.AEST_TZ)
        pd.DataFrame(
            {"datetime": times, "PM2.5_TEST SITE": range(len(times))}
        ).to_csv(self.csv_path, index=False)
        _FakeAQMS.requests = []
        _FakeAQMS.responses = []

    def tearDown(self):
        self.tempdir.cleanup()

    def _patch_dependencies(self):
        return mock.patch.multiple(
            updater,
            NSWAirQualityAPIClient=_FakeAQMS,
            _build_site_maps=mock.Mock(return_value=({updater._canon("TEST SITE"): 1}, {1: "TEST SITE"})),
            _available_parameter_codes=mock.Mock(return_value={"PM2.5"}),
        )

    def test_appends_only_six_missing_closed_hours_and_is_idempotent(self):
        _FakeAQMS.responses = [_Response(_observations_for_day())]
        now = pd.Timestamp("2026-07-10 17:35", tz=updater.AEST_TZ)

        with self._patch_dependencies():
            self.assertTrue(updater.update_nsw_region_input_file(self.csv_path, now=now))
            self.assertFalse(updater.update_nsw_region_input_file(self.csv_path, now=now))

        result = pd.read_csv(self.csv_path)
        result_times = pd.to_datetime(result["datetime"])
        self.assertEqual(len(result), 17)
        self.assertEqual(result_times.iloc[-1].hour, 16)
        self.assertEqual(result.loc[10, "PM2.5_TEST SITE"], 10)
        self.assertEqual(result.loc[11:, "PM2.5_TEST SITE"].tolist(), list(range(111, 117)))
        self.assertEqual(len(_FakeAQMS.requests), 1)

    def test_failed_batch_does_not_modify_existing_file(self):
        with open(self.csv_path, "rb") as source:
            before = source.read()
        _FakeAQMS.responses = [_Response([], status_code=503)]

        with self._patch_dependencies():
            with self.assertRaises(RuntimeError):
                updater.update_nsw_region_input_file(
                    self.csv_path,
                    now=pd.Timestamp("2026-07-10 17:35", tz=updater.AEST_TZ),
                )

        with open(self.csv_path, "rb") as source:
            self.assertEqual(source.read(), before)


if __name__ == "__main__":
    unittest.main()
