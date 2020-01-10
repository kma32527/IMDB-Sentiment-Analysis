import csv
import os
import random

# intiate lists
neg_list = list()
pos_list = list()

# open negative training sets and add them as strings onto negative list
for filename in os.listdir('./train/neg'):
    row=list()
    file = open('./train/neg/' + filename, encoding="utf8")
    row.append(file.read())
    row.append(0)
    neg_list.append(row)
    file.close()

# open positive training sets and add them as strings onto positive list
for filename in os.listdir('./train/pos'):
    row=list()
    file = open('./train/pos/' + filename, encoding="utf8")
    row.append(file.read())
    row.append(1)
    pos_list.append(row)
    file.close()

full_list = neg_list + pos_list
random.shuffle(full_list)
title=list()
title.append('text')
title.append('label')
with open('labeled_imdb.csv', mode='w', newline = '', encoding="utf8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(title)
    for row in full_list:
        writer.writerow(row)
print("# examples is " + str(len(full_list)))