class TPLOptions:
    """
    A container for options that control how a .TPl file should be handled when converting it to a set of object
    has_header_row
        A boolean indicating whether the file has a row containing header values. If True, that row wil be skipped when
        looking for data.
        Defaults to False

    """

    def __init__(self, header_line_numbers=0, split_expression="CATALOG", **kwargs):
        """

        """
        self.header_line_numbers = header_line_numbers
        self.file_extension = ".tpl"
        self.split_expression = split_expression
        self.tpl_dialect = kwargs
        self.columns_name = []

    def add_column(self, column):
        """

        """
        self.columns_name.append(column)