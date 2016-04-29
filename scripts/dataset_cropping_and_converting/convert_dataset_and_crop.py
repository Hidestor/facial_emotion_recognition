import csv
import numpy
from PIL import Image

emotionDictionary = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

def getImage(row):
    image = numpy.asarray(row, dtype=numpy.uint8)
    image = numpy.reshape(image,(48,48))
    img = Image.fromarray(image)
    img.show()

def convert_data_into_csv():
    csvfile = open('fer2013.csv')
    write_csvfile_train = open('dataset_train_new.csv','wb')
    write_csvfile_test = open('dataset_publictest_new.csv','wb')
    write_csvfile_test2 = open('dataset_privatetest_new.csv','wb')

    reader = csv.DictReader(csvfile)
    writer_train = csv.writer(write_csvfile_train, delimiter=',')
    writer_test = csv.writer(write_csvfile_test, delimiter=',')
    writer_test2 = csv.writer(write_csvfile_test2, delimiter=',')

    i =0 
    for row in reader:
    	emotion,pixels,usage= (row['emotion'], row['pixels'],row['Usage'])
    	pixels = map(int, pixels.split())
        if usage == 'Training':
            writer_train.writerow(pixels+ [int(emotion)])
            print i, 'Done.',usage
            if i == 28708:
                print 'Train dataset complete. Beginnning to create test dataset.'
        elif usage == 'PublicTest':
            print i, 'Done.',usage
            writer_test.writerow(pixels + [int(emotion)])
        elif usage == 'PrivateTest':
            print i, 'Done. ',usage
            writer_test2.writerow(pixels + [int(emotion)])


        i= i+1

convert_data_into_csv()

def load_data2():
    train_dataset = numpy.genfromtxt ('dataset_train_200.csv', delimiter=",")
    print 'Train dataset shape :',train_dataset.shape
    X = train_dataset[:,0:2304]
    y = train_dataset[:,2304:2305]
    # print y    
    print X.shape,y.shape
    test_dataset = numpy.genfromtxt ('dataset_test_100.csv', delimiter=",")
    print 'Train dataset shape :',test_dataset.shape



def load_data(ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)
    # Load the dataset
    train_dataset = numpy.genfromtxt ('dataset_train_200.csv', delimiter=",")
    test_dataset = numpy.genfromtxt ('dataset_test_100.csv', delimiter=",")

    # Convert data format
    def convert_data_format(data):
        X = data[:,0:2304]
        y = data[:,2304:2305]
        return (X,y)


    train_set = convert_data_format(train_dataset)
    test_set = convert_data_format(test_dataset)

    print 'Actual dataset shape train_set : ',train_set[0].shape
    print 'Actual dataset shape test_set : ',test_set[0].shape


    
    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)   
        print 'Getting a smaller dataset... new size of training + validation data ',train_set_len 
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//10):] for x in train_set]
    train_set = [x[:-(train_set_len//10)] for x in train_set]

    print 'Train set size : ',len(train_set[1])
    print 'Validation set size : ', len(valid_set[1])
    print 'Test set size : ',len(test_set[1])
    print 'Size of input layer (product of dimensions of image) : ',len(train_set[0][0])
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval
