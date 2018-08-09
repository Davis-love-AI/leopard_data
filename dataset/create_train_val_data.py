import os
path = 'F:/project/xml_tocsv/Annotations'
train_path = 'F:/project/xml_tocsv/data/train.txt'
val_path  ='F:/project/xml_tocsv/data/val.txt'
filenames = os.listdir(path)
length = 160
with open(train_path,'w') as fid:
   for i in range(len(filenames)):
       if i < length:
        fid.write(filenames[i].split('.')[0] + '\n')

with open(val_path,'w') as fid:
   for i in range(len(filenames)):
       if i >= length:
        fid.write(filenames[i].split('.')[0] + '\n')