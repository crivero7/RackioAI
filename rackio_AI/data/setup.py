import numpy
import platform


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('data', parent_package, top_path)
    config.add_data_dir('Leak')
    config.add_data_dir('pkl_files')
    if platform.python_implementation() != 'PyPy':
        config.add_extension('_svmlight_format_fast',
                             sources=['_svmlight_format_fast.pyx'],
                             include_dirs=[numpy.get_include()])
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())