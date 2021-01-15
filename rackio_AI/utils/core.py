from rackio_AI.core import RackioAI
from itertools import combinations as Combine


class Utils:
    """
    Documentation here
    """

    def __init__(self):
        """
        Documentation here
        """
        pass

    @staticmethod
    def check_default_kwargs(default_kw, kw):
        """
        Documentation here
        """
        kw = {key: kw[key] if key in kw.keys() else default_kw[key] for key in default_kw.keys()}
        
        return kw

    @staticmethod
    def get_column_names(df, **kwargs):
        """
        Documentation here
        pattern
        """
        default_kwargs = {
            "pattern": None
        }

        kw = Utils.check_default_kwargs(default_kwargs, kwargs)

        if not kw["pattern"]:
            
            return df.columns.to_list()

        else: 

            return

    @staticmethod
    def split_str(string: str, pattern: str, get_pos: int = 0):
        """
        Documentation here
        """
        return string.split(pattern)[get_pos]

    @staticmethod
    def get_combinations(columns=[], num=2):
        """
        Documentation here
        """
        return Combine(columns, num)

    @staticmethod
    def remove_row(iterable, loc):
        """
        Documentation here
        """
        iterable.pop(loc)
        
        return iterable
