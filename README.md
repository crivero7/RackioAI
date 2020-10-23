# RackioAI

**RackioAI** is a [Rackio](https://github.com/rack-io/rackio-framework) extension for Artificial Intelligence (AI) models.

The project was started in 2020 by [Carlos Rivero](https://github.com/crivero7) as a *Leak Detection System* and *Virtual Analyzer* project in **Intelcon System C.A** and [MCL Control S.A](http://mclcontrol.com/) respectively as a workaround to development and deployment *Deep Learning* models faster way.

# Installation

## Dependencies

**RackioAI** requieres:

* Python (>=3.8)
* numpy (1.18.5)
* scipy (1.4.1)
* scikit-learn (0.23.2)
* tensorflow (2.3.0)
* pandas (1.1.3)
* tqdm (4.50.2)
* Pillow (8.0.0)
* Rackio (0.9.7)

___

## User installation

The easiest way to install **RackioAI** is using pip
> pip install RackioAI

Then, to use it in any python project you can import it using:

```python
from rackio_AI import RackioAI
```

## User instantiation

The most important thing you keep in mind is that **RackioAI** is a [Rackio](https://github.com/rack-io/rackio-framework) extension, therefore, to use RackioAI in any project you must do the following steps, respecting the order
1. import Rackio
2. import RackioAI
3. to instantiate Rackio
4. do **RackioAI** callback with the Rackio object

see the following snippet code
```python
from rackio import Rackio
from rackio_AI import RackioAI

app = Rackio()

RackioAI(app)
```
Now, you can get access to **RackioAI** methods and attributes.
___
## Source code

You can check the latest sources with the command:

`git clone https://github.com/crivero7/RackioAI.git`
___
## Documentation (in progress...)

The RackioAI documentation can be found in [The RackioAI documentation can be found in](https://rackioai.readthedocs.io/en/latest/)

# Todo
* Changelog
* Contributing guide
* Testing code