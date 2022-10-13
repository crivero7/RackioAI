import unittest
import os
import re
from . import get_directory
from rackio_AI.readers.tpl import Genkey


genkey_test = {
    'Generated with OLGA version 2017.2.0': None,
    'Global keywords': {
        'OPTIONS': {
            'TEMPERATURE': 'WALL',
            'COMPOSITIONAL': 'OFF',
            'ELASTICWALLS': 'ON',
            'FLOWMODEL': 'OLGAHD'
        },
        'CASE': {
            'AUTHOR': 'Jesus E Varajas',
            'DATE': '02/09/2022',
            'PROJECT': 'Supe',
            'INFO': 'Modelo de parada, a partir del minuto 9 el sistema queda estable'
        },
        'FILES': {
            'PVTFILE': ("../../../07 Fluido/fase4.tab", "../../../07 Fluido/Fluido 0.tab")
        },
        'INTEGRATION': {
            'ENDTIME': {
                'VALUE': 3,
                'UNIT': 'M'
            },
            'MAXDT': {
                'VALUE': 0.1,
                'UNIT': 's'
            },
            'MINDT': {
                'VALUE': 0.03,
                'UNIT': 's'
            },
            'STARTTIME': {
                'VALUE': 0,
                'UNIT': 's'
            },
            'DTSTART': {
                'VALUE': 0.03,
                'UNIT': 's'
            }
        },
        'OUTPUT': {
            'WRITEFILE': 'OFF'
        },
        'TREND': {
            'DTPLOT': {
                'VALUE': 0.1,
                'UNIT': 's'
            }
        },
        'PROFILE': {
            'WRITEFILE': 'OFF',
            'DTPLOT': {
                'VALUE': 30,
                'UNIT': 's'
            },
            'DTTIME': {
                'VALUE': 0,
                'UNIT': 's'
            }
        },
        'RESTART': {
            'WRITE': 'OFF',
            'READFILE': 'OFF'
        },
        'BLACKOILCOMPONENT': [
            {
                'LABEL': 'oil',
                'TYPE': 'OIL',
                'OILSPECIFICGRAVITY': 0.872
            },
            {
                'LABEL': 'gas',
                'TYPE': 'GAS',
                'GASSPECIFICGRAVITY': 0.7
            }
        ],
        'BLACKOILFEED': {
            'LABEL': 'P500',
            'OILCOMPONENT': 'oil',
            'GASCOMPONENT': 'gas',
            'GOR': {
                'VALUE': 0,
                'UNIT': 'Sm3/Sm3'
            }
        },
        'BLACKOILOPTIONS': {
            'OILVISC-TUNING': 'ON',
            'GOR': {
                'VALUE': 0,
                'UNIT': 'Sm3/Sm3'
            },
            'GASSPECIFICGRAVITY': 0.7,
            'APIGRAVITY': 29.94,
            'OILVISC': {
                'VALUE': 4.839,
                'UNIT': 'CP'
            },
            'VISCTEMP': {
                'VALUE': 25,
                'UNIT': 'C'
            },
            'VISCPRESS': {
                'VALUE': 0,
                'UNIT': 'psig'
            }
        }
    },
    'Library keywords': {
        'WALL': {
            'LABEL': 'WALL-1',
            'THICKNESS': {
                'VALUES': (0.6, 3.315, 1.015),
                'UNIT': 'cm'
            },
            'MATERIAL': ('Fibra de vidrio', 'Concrete Coating HD', 'Stainless Steel'),
            'ELASTIC': 'ON'
        },
        'CENTPUMPCURVE': [
            {
                'LABEL': 'C-1',
                'VOLUMEFLOW': {
                    'VALUES': (0, 181.9067, 363.6619, 545.2656, 681.582, 817.8984, 954.2148, 1090.531),
                    'UNIT': 'm3/h'
                },
                'SPEED': {
                    'VALUES': (3299.76, 3299.76, 3299.76, 3299.76, 3299.76, 3299.76, 3299.76, 3299.76),
                    'UNIT': 'rpm'
                },
                'GVF': {
                    'VALUE': 0,
                    'UNIT': '%'
                },
                'DENSITY': {
                    'VALUE': 997,
                    'UNIT': 'kg/m3'
                },
                'EFFICIENCY': {
                    'VALUES': (63, 66.89, 69.22, 70, 69.56, 68.25, 66.06, 63),
                    'UNIT': '%'
                },
                'HEAD': {
                    'VALUES': (103.0491, 99.92642, 96.80372, 93.68102, 78.1651, 57.37963, 31.32459, 0),
                    'UNIT': 'm'
                }
            },
            {
                'LABEL': 'C-2',
                'VOLUMEFLOW': {
                    'VALUES': (0, 188.5352, 376.9134, 565.1346, 706.4182, 847.7018, 988.9855, 1130.269),
                    'UNIT': 'm3/h'
                },
                'SPEED': {
                    'VALUES': (3420, 3420, 3420, 3420, 3420, 3420, 3420, 3420),
                    'UNIT': 'rpm'
                },
                'GVF': {
                    'VALUE': 0,
                    'UNIT': '%'
                },
                'DENSITY': {
                    'VALUE': 997,
                    'UNIT': 'kg/m3'
                },
                'EFFICIENCY': {
                    'VALUES': (63, 66.89, 69.22, 70, 69.56, 68.25, 66.06, 63),
                    'UNIT': '%'
                },
                'HEAD': {
                    'VALUES': (110.696, 107.3415, 103.9871, 100.6327, 83.96541, 61.63753, 33.64906, 0),
                    'UNIT': 'm'
                }
            },
            {
                'LABEL': 'C-3',
                'VOLUMEFLOW': {
                    'VALUES': (0, 198.4581, 396.7509, 594.8785, 743.5981, 892.3177, 1041.037, 1189.757),
                    'UNIT': 'm3/h'
                },
                'SPEED': {
                    'VALUES': (3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600),
                    'UNIT': 'rpm'
                },
                'GVF': {
                    'VALUE': 0,
                    'UNIT': '%'
                },
                'DENSITY': {
                    'VALUE': 997,
                    'UNIT': 'kg/m3'
                },
                'EFFICIENCY': {
                    'VALUES': (63, 66.89, 69.22, 70, 69.56, 68.25, 66.06, 63),
                    'UNIT': '%'
                },
                'HEAD': {
                    'VALUES': (122.6548, 118.938, 115.2212, 111.5044, 93.03646, 68.29643, 37.28428, 0),
                    'UNIT': 'm'
                }
            }
        ],
        'MATERIAL': [
            {
                'LABEL': 'Stainless Steel',
                'CAPACITY': {
                    'VALUE': 450,
                    'UNIT': 'J/kg-C'
                },
                'CONDUCTIVITY': {
                    'VALUE': 20,
                    'UNIT': 'W/m-K'
                },
                'DENSITY': {
                    'VALUE': 7850,
                    'UNIT': 'kg/m3'
                },
                'EMOD': {
                    'VALUE': 210942150000,
                    'UNIT': 'Pa'
                }
            },
            {
                'LABEL': 'Fibra de vidrio',
                'CAPACITY': {
                    'VALUE': 450,
                    'UNIT': 'J/kg-C'
                },
                'CONDUCTIVITY': {
                    'VALUE': 20,
                    'UNIT': 'W/m-C'
                },
                'DENSITY': {
                    'VALUE': 7850,
                    'UNIT': 'kg/m3'
                },
                'EMOD': {
                    'VALUE': 45445704000,
                    'UNIT': 'Pa'
                }
            },
            {
                'LABEL': 'Concrete Coating HD',
                'CAPACITY': {
                    'VALUE': 880,
                    'UNIT': 'J/kg-C'
                },
                'CONDUCTIVITY': {
                    'VALUE': 2.7,
                    'UNIT': 'W/m-K'
                },
                'DENSITY': {
                    'VALUE': 3000,
                    'UNIT': 'kg/m3'
                },
                'EMOD': {
                    'VALUE': 50481503000,
                    'UNIT': 'Pa'
                }
            }
        ]
    },
    'Network Component': [
        {
            'NETWORKCOMPONENT': {
                'TYPE': 'FLOWPATH',
                'TAG': 'FLOWPATH_1'
            },
            'PARAMETERS': {
                'LABEL': 'Pipeline'
            },
            'BRANCH': {
                'FLUID': 'DIESEL'
            },
            'GEOMETRY': {
                'LABEL': 'GEOMETRY-1'
            },
            'PIPE': [
                {
                    'ROUGHNESS': {
                        'VALUE': 0.0053,
                        'UNIT': 'mm'
                    },
                    'LABEL': 'PIPE-1',
                    'WALL': 'WALL-1',
                    'NSEGMENT': 3,
                    'LSEGMENT': {
                        'VALUES': (0.96168, 0.989589, 1.04873),
                        'UNIT': 'm'
                    },
                    'LENGTH': {
                        'VALUE': 3,
                        'UNIT': 'm'
                    },
                    'ELEVATION': {
                        'VALUE': 0,
                        'UNIT': 'm'
                    },
                    'DIAMETER': {
                        'VALUE': 20.32,
                        'UNIT': 'cm'
                    }
                },
                {
                    'ROUGHNESS': {
                        'VALUE': 0.0053,
                        'UNIT': 'mm'
                    },
                    'LABEL': 'PIPE-2',
                    'WALL': 'WALL-1',
                    'NSEGMENT': 15,
                    'LSEGMENT': {
                        'VALUES': (0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.980903, 0.997747, 0.886458, 0.77429, 0.666334, 0.566139),
                        'UNIT': 'm'
                    },
                    'LENGTH': {
                        'VALUE': 13.7,
                        'UNIT': 'm'
                    },
                    'ELEVATION': {
                        'VALUE': 13.7,
                        'UNIT': 'm'
                    },
                    'DIAMETER': {
                        'VALUE': 20.32,
                        'UNIT': 'cm'
                    }
                },
            ],
            'TRENDDATA': [
                {
                    'ABSPOSITION': {
                        'VALUES': (19, 58, 1378),
                        'UNIT': 'm'
                    },
                    'VARIABLE': '(KAPPA, PT, TM)'
                },
                {
                    'ABSPOSITION': {
                        'VALUES': (19, 58, 1378),
                        'UNIT': 'm'
                    },
                    'VARIABLE': '(GT, ROHL)'
                },
                {
                    'CENTRIFUGALPUMP': 'PUMP',
                    'VARIABLE': 'PUMPSPEED'
                },
                {
                    'LEAK': 'LEAK',
                    'VARIABLE': '(ACMLK, GTLEAK, PTLKUP)'
                },
                {
                    'ABSPOSITION': {
                        'VALUES': (19, 58, 1378),
                        'UNIT': 'm'
                    },
                    'VARIABLE': '(SSP, VISHLTAB)'
                },
                {
                    'VALVE': ('V-out', 'V-in'),
                    'VARIABLE': '(PVALVE, VALVOP)'
                }
            ],
            'PROFILEDATA': {
                'VARIABLE': '(GT, PT, QOST, STDROHL, TM, VISHLTAB)'
            },
            'CENTRIFUGALPUMP': {
                'LABEL': 'PUMP',
                'MAXSPEED': {
                    'VALUE': 7200,
                    'UNIT': 'rpm'
                },
                'CURVEMODE': 'SINGLEPHASE',
                'ABSPOSITION': {
                    'VALUE': 1.5,
                    'UNIT': 'm'
                },
                'CURVES': ('C-1', 'C-2', 'C-3')
            },
            'HEATTRANSFER': [
                {
                    'LABEL': 'Air',
                    'PIPE': ('PIPE-1', 'PIPE-2', 'PIPE-3'),
                    'HOUTEROPTION': 'AIR',
                    'TAMBIENT': {
                        'VALUE': 21,
                        'UNIT': 'C'
                    }
                },
                {
                    'LABEL': 'Water',
                    'PIPE': 'PIPE-8',
                    'HOUTEROPTION': 'WATER',
                    'TAMBIENT': {
                        'VALUE': 21,
                        'UNIT': 'C'
                    }
                },
                {
                    'LABEL': 'Soil',
                    'PIPE': ('PIPE-9', 'PIPE-10'),
                    'HOUTEROPTION': 'HGIVEN',
                    'TAMBIENT': {
                        'VALUE': 21,
                        'UNIT': 'C'
                    },
                    'HAMBIENT': {
                        'VALUE': 10000,
                        'UNIT': 'W/m2-C'
                    }
                }
            ],
            'LEAK': {
                'LABEL': 'LEAK',
                'VALVETYPE': 'OLGAVALVE',
                'ABSPOSITION': {
                    'VALUE': 50,
                    'UNIT': 'm'
                },
                'TIME': {
                    'VALUE': 0,
                    'UNIT': 's'
                },
                'BACKPRESSURE': {
                    'VALUE': 0,
                    'UNIT': 'psig'
                },
                'DIAMETER': {
                    'VALUE': 0,
                    'UNIT': 'in'
                }
            },
            'VALVE': [
                {
                    'LABEL': 'C-1',
                    'MODEL': 'HYDROVALVE',
                    'ABSPOSITION': {
                        'VALUE': 16.7,
                        'UNIT': 'm'
                    },
                    'DIAMETER': {
                        'VALUE': 20.32,
                        'UNIT': 'cm'
                    }
                },
                {
                    'LABEL': 'V-out',
                    'MODEL': 'HYDROVALVE',
                    'TIME': {
                        'VALUE': 0,
                        'UNIT': 'M'
                    },
                    'STROKETIME': {
                        'VALUE': 0,
                        'UNIT': 's'
                    },
                    'ABSPOSITION': {
                        'VALUE': 1410,
                        'UNIT': 'm'
                    },
                    'DIAMETER': {
                        'VALUE': 20.32,
                        'UNIT': 'cm'
                    },
                    'OPENING': 0.13332
                }
            ],
            'CHECKVALVE': [
                {
                    'LABEL': 'CHECK-1',
                    'ABSPOSITION': {
                        'VALUE': 1492,
                        'UNIT': 'm'
                    }
                },
                {
                    'LABEL': 'CHECK-2',
                    'ABSPOSITION': {
                        'VALUE': 1,
                        'UNIT': 'm'
                    }
                }
            ]
        },
        {
            'NETWORKCOMPONENT': {
                'TYPE': 'MANUALCONTROLLER',
                'TAG': 'MANUALCONTROLLER_1'
            },
            'PARAMETERS': {
                'LABEL': 'Control-Pump',
                'TIME': {
                    'VALUE': 0,
                    'UNIT': 'M'
                },
                'SETPOINT': 0.4409,
                'MODE': 'AUTOMATIC',
                'OPENINGTIME': {
                    'VALUE': 10,
                    'UNIT': 's'
                },
                'CLOSINGTIME': {
                    'VALUE': 10,
                    'UNIT': 's'
                }
            },
            'TRENDDATA': {
                'VARIABLE': 'CONTR'
            }

        }
    ],
    'Connections': {
        'CONNECTION': [
            {
                'TERMINALS': ('FLOWPATH_1 INLET', 'NODE_1 FLOWTERM_1')
            },
            {
                'TERMINALS': ('MANUALCONTROLLER_1 CONTR_1', 'FLOWPATH_1 PUMP@SPEEDSIG')
            }
        ]
    }
}


class TestGenkey(unittest.TestCase):

    def setUp(self):
        """

        :return:
        """
        self.filename = os.path.join(
            get_directory('Leak'), 'genkey', '01.genkey')
        self.genkey = Genkey()
        self.genkey.read(filename=self.filename)

    def test_01_genkey_as_dict(self):
        """

        :return:
        """
        self.assertIsInstance(self.genkey, dict)

    def test_02_check_primary_keys(self):

        self.assertListEqual(list(self.genkey.keys()),
                             list(genkey_test.keys()))

    def test_03_check_global_keywords_as_dict(self):

        self.assertIsInstance(self.genkey['Global keywords'], dict)

    def test_04_check_global_keywords_keys(self):

        self.assertListEqual(sorted(list(self.genkey['Global keywords'].keys())), sorted(list(
            genkey_test['Global keywords'].keys())))

    def test_05_check_library_keywords_as_dict(self):

        self.assertIsInstance(self.genkey['Library keywords'], dict)

    def test_06_check_library_keywords_keys(self):

        self.assertListEqual(sorted(list(self.genkey['Library keywords'].keys())), sorted(list(
            genkey_test['Library keywords'].keys())))

    def test_07_check_network_component_as_list(self):
        
        self.assertIsInstance(self.genkey['Network Component'], list)

    def test_08_check_connections_as_dict(self):

        self.assertIsInstance(self.genkey['Connections'], dict)

    def test_09_check_global_keywords(self):

        with self.subTest("Testing Global Keywords - OPTIONS"):

            self.assertDictEqual(
                self.genkey['Global keywords']['OPTIONS'], genkey_test['Global keywords']['OPTIONS'])

        with self.subTest("Testing Global Keywords - CASE"):

            self.assertDictEqual(
                self.genkey['Global keywords']['CASE'], genkey_test['Global keywords']['CASE'])

        with self.subTest("Testing Global Keywords - FILES"):

            self.assertDictEqual(
                self.genkey['Global keywords']['FILES'], genkey_test['Global keywords']['FILES'])

        with self.subTest("Testing Global Keywords - INTEGRATION"):

            self.assertDictEqual(
                self.genkey['Global keywords']['INTEGRATION'], genkey_test['Global keywords']['INTEGRATION'])

        with self.subTest("Testing Global Keywords - OUTPUT"):

            self.assertDictEqual(
                self.genkey['Global keywords']['OUTPUT'], genkey_test['Global keywords']['OUTPUT'])

        with self.subTest("Testing Global Keywords - TREND"):

            self.assertDictEqual(
                self.genkey['Global keywords']['TREND'], genkey_test['Global keywords']['TREND'])

        with self.subTest("Testing Global Keywords - PROFILE"):

            self.assertDictEqual(
                self.genkey['Global keywords']['PROFILE'], genkey_test['Global keywords']['PROFILE'])

        with self.subTest("Testing Global Keywords - RESTART"):

            self.assertDictEqual(
                self.genkey['Global keywords']['RESTART'], genkey_test['Global keywords']['RESTART'])

        with self.subTest("Testing Global Keywords - BLACKOILCOMPONENT"):

            self.assertListEqual(self.genkey['Global keywords']['BLACKOILCOMPONENT'],
                                 genkey_test['Global keywords']['BLACKOILCOMPONENT'])

        with self.subTest("Testing Global Keywords - BLACKOILFEED"):

            self.assertDictEqual(
                self.genkey['Global keywords']['BLACKOILFEED'], genkey_test['Global keywords']['BLACKOILFEED'])

        with self.subTest("Testing Global Keywords - BLACKOILOPTIONS"):

            self.assertDictEqual(self.genkey['Global keywords']['BLACKOILOPTIONS'],
                                 genkey_test['Global keywords']['BLACKOILOPTIONS'])

    def test_10_check_library_keywords(self):

        with self.subTest("Testing Library Keywords - WALL"):

            self.assertDictEqual(
                self.genkey['Library keywords']['WALL'], genkey_test['Library keywords']['WALL'])

        with self.subTest("Testing Library Keywords - CENTPUMPCURVE"):

            self.assertListEqual(
                self.genkey['Library keywords']['CENTPUMPCURVE'], genkey_test['Library keywords']['CENTPUMPCURVE'])

        with self.subTest("Testing Library Keywords - MATERIAL"):

            self.assertListEqual(
                self.genkey['Library keywords']['MATERIAL'], genkey_test['Library keywords']['MATERIAL'])

    def test_11_check_network_component(self):

        components = self.genkey['Network Component']

        for elem, component in enumerate(components):

            with self.subTest("Testing Network Component"):

                self.assertDictEqual(component, genkey_test['Network Component'][elem])

    def test_12_check_connections(self):

        with self.subTest("Testing Connections - CONNECTION"):

            self.assertListEqual(
                self.genkey['Connections']['CONNECTION'], genkey_test['Connections']['CONNECTION'])
