from sklearn.feature_selection import RFE as rfe


class RFE(rfe):
    """
    Scikit learn RFE extension class
    """

    def __init__(self, estimator, ):
        """
        Documentation here
        """
        super(RFE, self).__init__(estimator)
