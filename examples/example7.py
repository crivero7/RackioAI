import os
from rackio_AI import RackioAI
from rackio import Rackio

app = Rackio()

RackioAI(app)

os.chdir('..')
cwd = os.getcwd()
filename = os.path.join(cwd, 'rackio_AI', 'data', 'tpl_files','Leak112.tpl')

RackioAI.load_data(filename)

df = RackioAI.convert_data_to('dataframe')
print(' ')
print('----------------------------------------------------------------------------------------')
print('DATAFRAME BEFORE PERSISTING')
print('----------------------------------------------------------------------------------------')
print(' ')
df.info()

filename = 'test'
# Save pkl object
RackioAI.save(df, filename)

# Load pkl object
df2 = RackioAI.load(filename)

print(' ')
print('----------------------------------------------------------------------------------------')
print('DATAFRAME AFTER PERSISTING')
print('----------------------------------------------------------------------------------------')
print(' ')
df2.info()