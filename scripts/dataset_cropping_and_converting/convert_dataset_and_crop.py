import csv
import numpy


def convert_data_into_csv():
    csvfile = open('fer2013.csv')
    write_csvfile_train = open('dataset_train.csv','wb')
    write_csvfile_test = open('dataset_test.csv','wb')

    reader = csv.DictReader(csvfile)
    writer_train = csv.writer(write_csvfile_train, delimiter=',')
    writer_test = csv.writer(write_csvfile_test, delimiter=',')

    i =0 
    for row in reader:
    	emotion,pixels,usage= (row['emotion'], row['pixels'],row['Usage'])
    	pixels = map(int, pixels.split())
        if usage == 'Training':
            writer_train.writerow(pixels+ [int(emotion)])
            print i, 'Done.',usage
            if i == 28708:
                print 'Train dataset complete. Beginnning to create test dataset.'
        else:
            print i, 'Done.',usage
            writer_test.writerow(pixels + [int(emotion)])
        i= i+1

convert_data_into_csv()

