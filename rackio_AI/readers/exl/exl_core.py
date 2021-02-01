import pandas as pd
from easy_deco.progress_bar import ProgressBar
from rackio_AI.utils import Utils
import rackio_AI
import os
from easy_deco.del_temp_attr import set_to_methods, del_temp_attr


@set_to_methods(del_temp_attr)
class EXL:
    """
    Supports xls, xlsx, xlsm, xlsb, odf, ods and odt file extensions read from a local filesystem or URL.
    Supports an option to read a single sheet or a list of sheets.
    """

    _instances = list()

    def __init__(self):

        EXL._instances.append(self)

    def read(self, pathname: str, **exl_options):
        """
        Read an Excel file into a pandas DataFrame.

        ### **Parameters**

        * **:param pathname: (str, path object, file-like object or directory path)**
        Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http,
        ftp, s3, and file. For file URLs, a host is expected.
        * **:param sheet_name: (str, int, list, or None, default 0)**
        Strings are used for sheet names. Integers are used in zero-indexed sheet positions. Lists of strings/integers are used to request multiple sheets. Specify None to get all sheets.
        Available cases:
            * Defaults to 0: 1st sheet as a *DataFrame*
            * 1: 2nd sheet as a *DataFrame*
            * "Sheet1": Load sheet with name “Sheet1”
            * [0, 1, "Sheet5"]: Load first, second and sheet named “Sheet5” as a dict of *DataFrame*
            * None: All sheets.
        * **:param header: (int, list of int, default 0)**
        Row (0-indexed) to use for the column labels of the parsed DataFrame. If a list of integers is passed
        those row positions will be combined into a MultiIndex. Use None if there is no header.
        * **:param names: (array-like, default None)**
        List of column names to use. If file contains no header row, then you should explicitly pass header=None.
        * **:param index_col: (int, list of int, default None)**
        Column (0-indexed) to use as the row labels of the DataFrame. Pass None if there is no such column. 
        If a list is passed, those columns will be combined into a MultiIndex. If a subset of data is 
        selected with usecols, index_col is based on the subset.
        * **:param usecols: (int, str, list-like, or callable default None)**
            * If None, then parse all columns.
            * If str, then indicates comma separated list of Excel column letters and column ranges 
            (e.g. “A:E” or “A,C,E:F”). Ranges are inclusive of both sides.
            * If list of int, then indicates list of column numbers to be parsed.
            * If list of string, then indicates list of column names to be parsed.
            * If callable, then evaluate each column name against it and parse the column if the callable returns True.
        Returns a subset of the columns according to behavior above.
        * **:param squeeze: (bool, default False)**
        If the parsed data only contains one column then return a Series.
        * **:param dtype: (Type name or dict of column -> type, default None)**
        Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32} Use object to preserve data as stored
        in Excel and not interpret dtype. If converters are specified, they will be applied INSTEAD of dtype conversion.
        * **:param mangle_dupe_cols: (bool, default True)**
        Duplicate columns will be specified as ‘X’, ‘X.1’, …’X.N’, rather than ‘X’…’X’. Passing in False will cause 
        data to be overwritten if there are duplicate names in the columns.
        * **:param engine: (str, default None)**
        If io is not a buffer or path, this must be set to identify io. Supported engines: 
        “xlrd”, “openpyxl”, “odf”, “pyxlsb”. Engine compatibility :
            * “xlrd” supports old-style Excel files (.xls).
            * “openpyxl” supports newer Excel file formats.
            * “odf” supports OpenDocument file formats (.odf, .ods, .odt).
            * “pyxlsb” supports Binary Excel files.
        Changed in version 1.2.0: The engine xlrd now only supports old-style .xls files. 
        When engine=None, the following logic will be used to determine the engine:
            * If path_or_buffer is an OpenDocument format (.odf, .ods, .odt), then odf will be used.
            * Otherwise if path_or_buffer is an xls format, xlrd will be used.
            * Otherwise if openpyxl is installed, then openpyxl will be used.
            * Otherwise if xlrd >= 2.0 is installed, a ValueError will be raised.
            * Otherwise xlrd will be used and a FutureWarning will be raised. This case will raise a 
        ValueError in a future version of pandas.
        * **:param converters: (dict, default None)**
        Dict of functions for converting values in certain columns. Keys can either be integers or 
        column labels, values are functions that take one input argument, the Excel cell content, and 
        return the transformed content.
        * **:param true_values: (list, default None)**
        Values to consider as True.
        * **:param false_values: (list, default None)**
        Values to consider as False.
        * **:param skiprows: (list-like, int, or callable, optional)**
        Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file. 
        If callable, the callable function will be evaluated against the row indices, returning True 
        if the row should be skipped and False otherwise. An example of a valid callable argument 
        would be lambda x: x in [0, 2].
        * **:param nrows: (int, default None)**
        Number of rows to parse.
        * **:param na_values: (scalar, str, list-like, or dict, default None)**
        Additional strings to recognize as NA/NaN. If dict passed, specific per-column NA values. 
        By default the following values are interpreted as NaN: ‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, 
        ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, ‘NA’, ‘NULL’, 
        ‘NaN’, ‘n/a’, ‘nan’, ‘null’.
        * **:param keep_default_na: (bool, default True)**
        Whether or not to include the default NaN values when parsing the data. Depending on whether 
        na_values is passed in, the behavior is as follows:
            * If keep_default_na is True, and na_values are specified, na_values is appended to the 
        default NaN values used for parsing.
            * If keep_default_na is True, and na_values are not specified, only the default NaN 
        values are used for parsing.
            * If keep_default_na is False, and na_values are specified, only the NaN values specified 
        na_values are used for parsing.
            * If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.
        Note that if na_filter is passed in as False, the keep_default_na and na_values parameters will 
        be ignored.
        * **:param na_filter: (bool, default True)**
        Detect missing value markers (empty strings and the value of na_values). In data without any NAs, 
        passing na_filter=False can improve the performance of reading a large file.
        * **:param verbose: (bool, default False)**
        Indicate number of NA values placed in non-numeric columns.
        * **:param parse_dates: (bool, list-like, or dict, default False)**
        The behavior is as follows:
            * bool. If True -> try parsing the index.
            * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
            * list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.
            * dict, e.g. {‘foo’ : [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’
        If a column or index contains an unparseable date, the entire column or index will be returned 
        unaltered as an object data type. If you don`t want to parse some cells as date just change their 
        type in Excel to “Text”. For non-standard datetime parsing, use pd.to_datetime after pd.read_excel.
        Note: A fast-path exists for iso8601-formatted dates.
        * **:param date_parser: (function, optional)**
        Function to use for converting a sequence of string columns to an array of datetime instances. 
        The default uses dateutil.parser.parser to do the conversion. Pandas will try to call date_parser 
        in three different ways, advancing to the next if an exception occurs: 1) Pass one or more arrays 
        (as defined by parse_dates) as arguments; 2) concatenate (row-wise) the string values from the columns 
        defined by parse_dates into a single array and pass that; and 3) call date_parser once for each row using 
        one or more strings (corresponding to the columns defined by parse_dates) as arguments.
        * **:param thousands: (str, default None)**
        Thousands separator for parsing string columns to numeric. Note that this parameter is only necessary 
        for columns stored as TEXT in Excel, any numeric columns will automatically be parsed, regardless 
        of display format.
        * **:param decimal: (str, default ‘.’)**
        Character to recognize as decimal point (e.g. use ‘,’ for European data).
        * **:param skipfooter: (int, default 0)**
        Rows at the end to skip (0-indexed).
        Character to break file into lines. Only valid with C parser.
        * **:param comment: (str, default None)**
        Comments out remainder of line. Pass a character or characters to this argument to indicate comments 
        in the input file. Any data between the comment string and the end of the current line is ignored.
        * **:param mangle_dupe_cols: (bool, default True)**
        Duplicate columns will be specified as ‘X’, ‘X.1’, …’X.N’, rather than ‘X’…’X’. Passing in False will 
        cause data to be overwritten if there are duplicate names in the columns.
        * **:param storage_options: (dict, optional)** 
        Extra options that make sense for a particular storage connection, e.g. host, port, username, password,
        etc., if using a URL that will be parsed by fsspec, e.g., starting “s3://”, “gcs://”. An error will 
        be raised if providing this argument with a local path or a file-like buffer. See the fsspec and backend 
        storage implementation docs for the set of allowed keys and values

        ### **Returns**

        * **(DataFrame or dict of DataFrames)** *

        DataFrame from the passed in Excel file. See notes in sheet_name argument for more information on 
        when a dict of DataFrames is returned.
    
        _______
        ### **Snippet Code**

        ```python
        >>> import os
        >>> from rackio_AI import RackioAI, get_directory
        >>> directory = os.path.join(get_directory('excel'))
        >>> RackioAI.load(directory, ext=".xlsx", header=0, sheet_name="SalesOrders")
            OrderDate   Region       Rep     Item  Units  Unit Cost    Total
        0  2019-01-06     East     Jones   Pencil     95       1.99   189.05
        1  2019-01-23  Central    Kivell   Binder     50      19.99   999.50
        2  2019-02-09  Central   Jardine   Pencil     36       4.99   179.64
        3  2019-02-26  Central      Gill      Pen     27      19.99   539.73
        4  2019-03-15     West   Sorvino   Pencil     56       2.99   167.44
        5  2019-04-01     East     Jones   Binder     60       4.99   299.40
        6  2019-04-18  Central   Andrews   Pencil     75       1.99   149.25
        7  2019-05-05  Central   Jardine   Pencil     90       4.99   449.10
        8  2019-05-22     West  Thompson   Pencil     32       1.99    63.68
        9  2019-06-08     East     Jones   Binder     60       8.99   539.40
        10 2019-06-25  Central    Morgan   Pencil     90       4.99   449.10
        11 2019-07-12     East    Howard   Binder     29       1.99    57.71
        12 2019-07-29     East    Parent   Binder     81      19.99  1619.19
        13 2019-08-15     East     Jones   Pencil     35       4.99   174.65
        14 2019-09-01  Central     Smith     Desk      2     125.00   250.00
        15 2019-09-18     East     Jones  Pen Set     16      15.99   255.84
        16 2019-10-05  Central    Morgan   Binder     28       8.99   251.72
        17 2019-10-22     East     Jones      Pen     64       8.99   575.36
        18 2019-11-08     East    Parent      Pen     15      19.99   299.85
        19 2019-11-25  Central    Kivell  Pen Set     96       4.99   479.04
        20 2019-12-12  Central     Smith   Pencil     67       1.29    86.43
        21 2019-12-29     East    Parent  Pen Set     74      15.99  1183.26
        22 2020-01-15  Central      Gill   Binder     46       8.99   413.54
        23 2020-02-01  Central     Smith   Binder     87      15.00  1305.00
        24 2020-02-18     East     Jones   Binder      4       4.99    19.96
        25 2020-03-07     West   Sorvino   Binder      7      19.99   139.93
        26 2020-03-24  Central   Jardine  Pen Set     50       4.99   249.50
        27 2020-04-10  Central   Andrews   Pencil     66       1.99   131.34
        28 2020-04-27     East    Howard      Pen     96       4.99   479.04
        29 2020-05-14  Central      Gill   Pencil     53       1.29    68.37
        30 2020-05-31  Central      Gill   Binder     80       8.99   719.20
        31 2020-06-17  Central    Kivell     Desk      5     125.00   625.00
        32 2020-07-04     East     Jones  Pen Set     62       4.99   309.38
        33 2020-07-21  Central    Morgan  Pen Set     55      12.49   686.95
        34 2020-08-07  Central    Kivell  Pen Set     42      23.95  1005.90
        35 2020-08-24     West   Sorvino     Desk      3     275.00   825.00
        36 2020-09-10  Central      Gill   Pencil      7       1.29     9.03
        37 2020-09-27     West   Sorvino      Pen     76       1.99   151.24
        38 2020-10-14     West  Thompson   Binder     57      19.99  1139.43
        39 2020-10-31  Central   Andrews   Pencil     14       1.29    18.06
        40 2020-11-17  Central   Jardine   Binder     11       4.99    54.89
        41 2020-12-04  Central   Jardine   Binder     94      19.99  1879.06
        42 2020-12-21  Central   Andrews   Binder     28       4.99   139.72
        
        ```
        """
        json_dir = os.path.join(rackio_AI.__file__.replace(os.path.join(os.path.sep, '__init__.py'), ''), 'readers', 'exl', 'json')
        default_exl_options = Utils.load_json(os.path.join(json_dir, "exl_options.json"))
        options = Utils.check_default_kwargs(default_exl_options, exl_options)

        self._df_ = list()
        self.__read(pathname, **options)
        df = pd.concat(self._df_)
            
        return df

    @ProgressBar(desc="Reading excel files...", unit="file")
    def __read(self, excel_file, **kwargs):
        """
        Decorated function to visualize the progress bar during the execution of *read*
        method in the pipeline

        **Parameters**

        * **:param iterable:** (list)

        **returns**

        None
        """
            
        self._df_.append(pd.read_excel(excel_file, **kwargs))
        
        return
    

if __name__ == "__main__":
    import doctest

    doctest.testmod()