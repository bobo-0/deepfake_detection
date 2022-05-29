import os
o_train = '../../../data/ori/train'
o_test = '../../../data/ori/test'
h_train = '../../../data/hti/train'
h_test = '../../../data/hti/test'
r_train = '../../../data/rti/train'
r_test = '../../../data/rti/test'

def get_files_count(folder_path):
    dirListing = os.listdir(folder_path)
    return len(dirListing)

print("ORI DATASET")
print(get_files_count(o_train+ '/real'))
print(get_files_count(o_train+ '/fake'))
print(get_files_count(o_test + '/real'))
print(get_files_count(o_test+ '/fake'))


print("HTI DATASET")
print(get_files_count(h_train+ '/real'))
print(get_files_count(h_train+ '/fake'))
print(get_files_count(h_test + '/real'))
print(get_files_count(h_test+ '/fake'))

print("RTI DATASET")
print(get_files_count(r_train+ '/real'))
print(get_files_count(r_train+ '/fake'))
print(get_files_count(r_test + '/real'))
print(get_files_count(r_test+ '/fake')) 

