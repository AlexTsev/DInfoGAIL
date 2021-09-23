import numpy as np
import pandas as pd

indexing_csv = pd.read_csv('./test_equal.csv')
index_old = indexing_csv[['index_old']]
index_new = indexing_csv[['index']]
print('ID')
print(index_old)
print(index_new)
index_old = np.asarray(index_old).tolist()
index_new = np.asarray(index_new).tolist()

#print(indexing_csv)
names_index_old = index_old
names_index_new = index_new


for i,y in zip(names_index_old, names_index_new):
    if(i==y):
        print('index_old:', i, 'index_new:', y)
        print('EQUAL')
    else:
        print('NOT-EQUAL')
        print('index_old:', i, 'index_new:', y)




