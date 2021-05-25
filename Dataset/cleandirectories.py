import os

dir_a = []
dir_b = []
for fileA in os.listdir('path_to_dir_a'):
    dir_a.append[fileA.split('.')[0]]
for fileB in os.listdir('path_to_dir_a'):
    dir_b.append[fileB.split('.')[0]]

for fileA in dir_a:
    if not fileA in dir_b:
        os.remove(os.path.join('path_to_dir_a', (fileA + '.jpg')))
