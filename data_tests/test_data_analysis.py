import pytest
import logging as log
import pandas as pd

from data_analysis import normalize_grades_column, convert_strings_to_numbers

DEFAULT_SERIES = pd.Series([1, 2, 3, 4])
DEFAULT_SERIES_TEXT = pd.Series(['1', '2', '<1'])


class test_normalize_grades_column:

    @staticmethod
    def positive():

        series = normalize_grades_column(DEFAULT_SERIES, 0, 4)
        assert series[0] == 25.0
        assert series[1] == 50.0
        assert series[2] == 75.0
        assert series[3] == 100.0

    @staticmethod
    def negative_small_limit():
        try:
            normalize_grades_column(DEFAULT_SERIES, 0, 0)
        except ArithmeticError:
            pass
        else:
            assert False, 'normalize_grades_column cannot get a limit smaller than 1'

    @staticmethod
    def negative_limit_below_base():
        try:
            normalize_grades_column(DEFAULT_SERIES, 3, 1)
        except ArithmeticError:
            pass
        else:
            assert False, 'Limit cannot be smaller than base'


class test_convert_strings_to_numbers:
    @staticmethod
    def positive():
        series = convert_strings_to_numbers(DEFAULT_SERIES_TEXT)
        assert series[0] == 1
        assert series[1] == 2
        assert series[2] == 0

    @staticmethod
    def negative_non_numeric_string():
        series = pd.Series(['a'])
        try:
            convert_strings_to_numbers(series)
        except ValueError:
            pass
        else:
            assert False, 'convert_strings_to_numbers should not be able to handle a string'