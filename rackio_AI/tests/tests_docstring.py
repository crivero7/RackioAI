import unittest
import doctest
import rackio_AI


def load_tests(loader, tests, ignore):

    # Main
    tests.addTests(doctest.DocTestSuite(rackio_AI.core))

    # Data Analysis
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.data_analysis_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.outliers))
    tests.addTests(doctest.DocTestSuite(rackio_AI.data_analysis.noise))

    # Readers
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.readers_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.tpl.tpl_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.pkl.pkl_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers.exl.exl_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.readers._csv_.csv_core))

    # Preprocessing
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.preprocessing_core))
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.feature_extraction))
    # tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.feature_selection))
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.kalman_filter))
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.scaler))
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.splitter))
    tests.addTests(doctest.DocTestSuite(rackio_AI.preprocessing.synthetic_data))

    # Pipeline
    tests.addTests(doctest.DocTestSuite(rackio_AI.pipeline.pipeline_core))

    # Managers
    tests.addTests(doctest.DocTestSuite(rackio_AI.managers.managers_core))

    # Models
    # tests.addTests(doctest.DocTestSuite(rackio_AI.models.models_core))

    # Utils
    tests.addTests(doctest.DocTestSuite(rackio_AI.utils.utils_core))

    return tests

if __name__ == '__main__':
    unittest.main()