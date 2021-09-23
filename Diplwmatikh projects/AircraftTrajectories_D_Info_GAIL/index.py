import numpy as np
import pandas as pd

indexing_csv = pd.read_csv('./dataset/train_set2-259040.csv')
obs_train_df = indexing_csv[['trajectory_ID']]
print('ID')
print(obs_train_df)
indexing_csv = np.asarray(obs_train_df).tolist()

#print(indexing_csv)
names = indexing_csv
temp = names[0]
#print(names)
count=0

counter=0
index_list=[]
for i in names:
    counter+=1
    print('counter:', counter)
    if(i==temp):
        print(i)
        count+=1
        print('equal_count:',count)
    if(i!=temp):
        index_list.append(count)
        temp = i
        print('temp:', temp)
        count=1
index_list.append('740')
print(index_list)

for b in index_list:
    print('', b)
print('len:', len(index_list))

index_pd = pd.DataFrame(index_list, columns=['index'])
index_pd.to_csv('./aviation-indexing-259040.csv', index=False)

#print(names[-1])




