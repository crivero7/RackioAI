import unittest
import doctest
import rackio_AI
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


    
def load_tests(loader, tests, ignore):
    
    tests.addTests(doctest.DocTestSuite(rackio_AI.core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.data_analysis_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.outliers))
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.noise))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.readers_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.tpl.tpl_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.pkl.pkl_core))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.readers.exl.exl_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers._csv_.csv_core))

    # Preprocessing
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.preprocessing_core))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.feature_extraction))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.feature_selection))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.kalman_filter))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.scaler))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.splitter))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.synthetic_data))
    
    return tests

if __name__ == '__main__':
    unittest.main()