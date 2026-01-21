import pandas as pd

from data_analysis import convert_strings_to_numbers, series_to_z_score, map_isco_score
from consts import ISCO_MAPPING

DEFAULT_SERIES = pd.Series([1, 2, 3, 4])
DEFAULT_SERIES_TEXT = pd.Series(['1', '2', '<1'])


class TestConvertStringsToNumbers:
    @staticmethod
    def test_positive():
        series = convert_strings_to_numbers(DEFAULT_SERIES_TEXT)
        assert series[0] == 1
        assert series[1] == 2
        assert series[2] == 0

    @staticmethod
    def test_negative_non_numeric_string():
        series = pd.Series(['a'])
        try:
            convert_strings_to_numbers(series)
        except ValueError:
            pass
        else:
            assert False, 'convert_strings_to_numbers should not be able to handle a string'


class TestSeriesToZScore:
    @staticmethod
    def test_positive():
        result = series_to_z_score(DEFAULT_SERIES, 2.5, 1)
        assert result[0] == -1.5
        assert result[1] == -0.5
        assert result[2] == 0.5
        assert result[3] == 1.5

    @staticmethod
    def test_negative_zero_std():

        try:
            series_to_z_score(DEFAULT_SERIES, 2.5, 0)
        except ZeroDivisionError:
            pass
        else:
            assert False, 'series_to_z_score should not be able to handle a std that equals 0'


class TestMapIscoScore:
    @staticmethod
    def test_positive():
        from numpy import nan
        for item in [' UNEMPLOYED ', nan, None]:
            assert map_isco_score(item) is nan
        assert map_isco_score(' SELF-EMPLOYED ') == 65
        for key in ISCO_MAPPING:
            assert map_isco_score(key) == ISCO_MAPPING[key]
        assert map_isco_score('1123412345aa') == 85

    @staticmethod
    def test_negative_non_existing_job():
        from numpy import nan
        assert map_isco_score('99') is nan
        assert map_isco_score('aa') is nan
