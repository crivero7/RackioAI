from rackio_AI.core import RackioAI


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
    def get_column_names(df):
        """
        Documentation here
        """
        return df.columns.to_list()

    @staticmethod
    def split_str(string: str, pattern: str, get_pos: int = 0):
        """
        Documentation here
        """
        return string.split(pattern)[get_pos]