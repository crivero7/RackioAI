# Installation

## Dependencies
**RackioAI** requieres:

* Python (>=3.8)
* pyxlsb==1.0.8
* odfpy==1.4.1
* openpyxl==3.0.6
* xlrd==1.2.0
* seaborn==0.11.1
* scikit-learn==0.24.1
* statsmodels==0.12.1
* easy-deco==0.1.2
* Pillow==8.0.0
* tensorflow==2.4.1
* PyWavelets==1.1.1)

The easiest way to install **RackioAI** is using pip

> pip install RackioAI

# Overview
Then, to use it in any python project you can import it using:

```python
from rackio_AI import RackioAI
```

## User instantiation
The most important thing you keep in mind is that **RackioAI** is a [Rackio](https://github.com/rack-io/rackio-framework)
 extension, therefore, to use RackioAI in any project you must do the following steps, respecting the order
 
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