"""
This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import json
import numpy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import theano
import theano.tensor as T
import cv2
import time
from six.moves import cPickle

from utils import shared_dataset, load_data , getImage
from nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

from theano.tensor.shared_randomstreams import RandomStreams
emotionDictionary = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

def noise_injection(train_set,std = 0.01,noise_distribution = 'normal'):
    '''Used for regularization. Noise is injected in the input '''
    srng = RandomStreams()
    number_of_examples = len(train_set)
    number_of_features = len(train_set[0])


    if noise_distribution == 'normal':
        print ' Normal noise added to the inputs.'
        noise_matrix = srng.normal(size = (number_of_examples,number_of_features),avg = 0.0,std = std)
    elif noise_distribution == 'uniform':
        print ' Uniform noise added to the inputs. ' 
        noise_matrix = srng.uniform(size = (number_of_examples,number_of_features),low = 0.0,high = std)
    else:
        noise_matrix = numpy.zeros((number_of_examples,number_of_features))
    
    f = theano.function([],noise_matrix)
    noise_matrix = f()
    train_set = train_set + noise_matrix        
    return train_set

def translate_image(data,move = 'top',steps = 1):
    '''
    Translates the image #steps to (right,left,top or bottom)
    '''

    translated_data = []
    for X in data : 
        if move == 'left':
            for step in range(steps):
                for index in range(step,3072,32):
                    X[index] = 0
            X = numpy.roll(X,-steps)        
        elif move == 'right':
            for step in range(steps):
                for index in range(31-step,3072,32):
                    X[index] = 0
            X = numpy.roll(X,steps)        
        elif move == 'top':
            for index in range(0,32*steps):
                X[index] = 0
            for index in range(1024,1024+32*steps):
                X[index] = 0
            for index in range(2048,2048+32*steps):
                X[index] = 0
            X = numpy.roll(X,-32*steps)
        elif move == 'bottom':
            for index in range(1024-32*steps,1024):
                X[index] = 0
            for index in range(2048-32*steps,2048):
                X[index] = 0
            for index in range(3072 -32*steps,3072):
                X[index] = 0
            X = numpy.roll(X,32*steps)
        translated_data.append(X)
    return translated_data

def predict_webcam_image(classifier,x,y,timegap = .1):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #resize the image to 48x48 
        res = cv2.resize(gray,(48,48), interpolation = cv2.INTER_CUBIC)

        # Display the resulting frames
        cv2.imshow('frame',gray)
        cv2.imshow('resized',res)

        input_image = numpy.reshape(res,(1,48*48))

        #Close the frame when 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        


        res = res/255
        temp = numpy.asarray([1])
        input_image_x,input_image_y = shared_dataset([input_image,temp])
        test_input_example = theano.function(
            inputs= [],
            outputs=[classifier.errors(y),classifier.p_y_given_x],
            givens={
                x: input_image_x,
                y: input_image_y
                }
            )

        error, input_image_output_probilities = test_input_example()

        print 'probabilities : ',input_image_output_probilities
        print 'Number predicted :', emotionDictionary[numpy.argmax(input_image_output_probilities)]
        print 'with probability :',numpy.max(input_image_output_probilities)*100
        print '-------------------------------------'
        
        time.sleep(timegap)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def save_model(classifier,n_hidden,n_hiddenLayers,filename):
    '''Saves the model as a json object by saving it's parameters.
    Pickle throwing some error.
    '''

    f = open('%s.save'%filename, 'wb')    
    f2 = open('%s_json.save'%filename,'wb') 
    parameters = []
    p1 = []
    for parameter in classifier.params:
        p1.append(parameter.eval())
        parameters.append(parameter.eval().tolist())

    parameters = parameters + [n_hidden,n_hiddenLayers]
    cPickle.dump(p1+[n_hidden,n_hiddenLayers] ,f)
    encode = json.dumps({'p':parameters})
    f2.write(encode)
    f.close()
    f2.close()

def load_model(filename):
    f = open('%s.save'%filename, 'rb') 
    filecontent= f.read()
    loaded_obj = json.loads(filecontent)
    f.close()
    print 'loaded'
    return loaded_obj['p']

def load_model2(filename):
    f = open('%s.save'%filename, 'rb') 
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

'''
def predict_from_trained_model2(filename):
    loaded_obj = load_model2(filename)
    n_hidden= loaded_obj[-2]
    n_hiddenLayers = loaded_obj[-1]
    parameters_shared = loaded_obj[0:-2]
    print parameters_shared
    # parameters = []
    # for parameter in parameters_shared:
    #     parameters.append(parameter.eval())
    parameters = parameters_shared


    # x = T.matrix('x')  # the data is presented as rasterized images
    # y = T.ivector('y')  # the labels are presented as 1D vector of

    # rng = numpy.random.RandomState(1234)

    # classifier = myMLP(rng, input = x, n_in =2304 , n_hidden = n_hidden, n_out= 7, n_hiddenLayers= n_hiddenLayers,parameters =parameters)

    # predict_webcam_image(classifier,x,y)
    # print 'Done...'

        # print parameters[-1].shape
    # parameters = numpy.array(parameters)
'''

def predict_from_trained_model3(filename):
    loaded_obj = load_model(filename)
    
    n_hidden= loaded_obj[-2]
    n_hiddenLayers = loaded_obj[-1]
    parameters_shared = loaded_obj[0:-2]
    parameters = []

    for p in parameters_shared:
        p1 = []
        for row in p:
            p1.append(numpy.asarray(row))
        parameters.append(numpy.asarray(p1))

    x = T.matrix('x')  
    y = T.ivector('y') 

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(rng, input = x, n_in =2304 , n_hidden = n_hidden, n_out= 7, n_hiddenLayers= n_hiddenLayers,parameters =parameters)
    predict_webcam_image(classifier,x,y)
    
    print 'Done...'


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False, smaller_set=True,timegap = 0.5,):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    train_set, valid_set, test_set = load_data(theano_shared=False)

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    print test_set_y.eval().shape
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    print 'n_train_batches : ',n_train_batches
    print 'n_valid_batches : ',n_valid_batches
    print 'n_test_batches : ',n_test_batches

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    # input, n_in, n_hidden, n_out, n_hiddenLayers
    classifier = myMLP(rng, input = x, n_in =2304 , n_hidden = n_hidden, n_out= 7, n_hiddenLayers= n_hiddenLayers,parameters =None)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)


    print ('MODEL TRAINED..')
    
    ##########
    # SAVE the MODEL
    ##########

    save_model(classifier,n_hidden,n_hiddenLayers,'mlp_classifier_nhidden_%s_hiddenlayers_%s_batchSize_%s_epochs_%s'%(n_hidden,n_hiddenLayers,batch_size,n_epochs))

    print 'Model Saved. '
    # load_model('mlp_classifier_nhidden_%s_hiddenlayers_%s_batchSize_%s_epochs_%s'%(n_hidden,n_hiddenLayers,batch_size,epochs))
    ###########
    # MAKE PREDICTION on NEW IMAGE:
    ###########
    # predict_webcam_image(classifier,x,y)

def test_noise_inject_at_input(learning_rate=0.01, L1_reg=0.00,L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False,std=0.01,noise_distribution = 'normal'):
    """
    Wrapper function for experiment of noise injection at input

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[0] = numpy.tile(train_set[0], [5,1])
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: injection noise to the training set
    # call noise_injection() ...
    train_set[0] = noise_injection(train_set[0],std,noise_distribution)

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model ')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training with noise injected inputs')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def test_noise_injection_at_weight(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False,std = 0.01):
    """
    Wrapper function for experiment of noise injection at weights

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset
    datasets = load_data(ds_rate=5)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]


    updates = []
    srng = RandomStreams()

    i = 0
    for param, gparam in zip(classifier.params, gparams):
        if i%2 == 0:
            updates.append((param, param - learning_rate * gparam + srng.normal(size = param.shape, avg = 0.0,std = std)))
        else:
            updates.append((param, param - learning_rate * gparam))
        i= i+1


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def test_data_augmentation(learning_rate=0.01,L1_reg=0.00, L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False,steps = 1):
    """
    Wrapper function for experiment of data augmentation

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """
    rng = numpy.random.RandomState(23455)

    # Load down-sampled dataset in raw format (numpy.darray, not Theano.shared)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), where each row
    # corresponds to an example. target is a numpy.ndarray of 1 dimension
    # (vector) that has the same length as the number of rows in the input.

    # Load the smaller dataset in raw Format, since we need to preprocess it
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    # Repeat the training set 5 times
    train_set[1] = numpy.tile(train_set[1], 5)

    # TODO: translate the dataset
    train_set_x_u = translate_image(train_set[0],'top',steps)
    train_set_x_d = translate_image(train_set[0],'bottom',steps)
    train_set_x_r = translate_image(train_set[0],'right',steps)
    train_set_x_l = translate_image(train_set[0],'left',steps)

    # Stack the original dataset and the synthesized datasets
    train_set[0] = numpy.vstack((train_set[0],
                       train_set_x_u,
                       train_set_x_d,
                       train_set_x_r,
                       train_set_x_l))

    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=32*32*3,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

def test_mlp_with_new_functionality(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False, smaller_set=True,example_index = 0,adversarial_parameter = 0.01,distribution = 'constant'):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """

    # load the dataset; download the dataset if it is not present
    train_set, valid_set, test_set = load_data(ds_rate=5, theano_shared=False)

    example_x = test_set[0][example_index:example_index+1]
    example_y = test_set[1][example_index:example_index+1]

    # example_x_reshape = numpy.reshape(example_x,(len(example_x),1))
    # example_y_reshape = numpy.reshape(example_y,(1,1))

    shared_example_x,shared_example_y = shared_dataset([example_x,example_y])
    # shared_example_x = shared_example_x.reshape(shared_example_x.reshape[0],-1)

    # shared_example_x = theano.shared(type =theano.tensor.matrix,value = numpy.asarray(example_x,dtype=theano.config.floatX),borrow = True)

    # shared_example_y  = theano.shared(type = theano.tensor.vector, value = numpy.asarray(example_y,dtype=theano.config.floatX),borrow = True)
    # Convert raw dataset to Theano shared variables.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # print ' shapes of the shared_examples', shared_example_y,shared_example_x

    print 'n_train_batches : ',n_train_batches
    print 'n_valid_batches : ',n_valid_batches
    print 'n_test_batches : ',n_test_batches
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    # input, n_in, n_hidden, n_out, n_hiddenLayers
    classifier = myMLP(rng, input = x, n_in =3072 , n_hidden = n_hidden, n_out= 10, n_hiddenLayers= n_hiddenLayers)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


    test_example = theano.function(
        inputs= [index],
        outputs=[classifier.errors(y),classifier.p_y_given_x],
        givens={
            x: test_set_x[index * 1:(index + 1) * 1],
            y: test_set_y[index * 1:(index + 1) * 1]
            }
        )

    test_example2 = theano.function(
        inputs= [],
        outputs=[classifier.errors(y),classifier.p_y_given_x],
        givens={
            x: shared_example_x,
            y: shared_example_y
            }
        )



    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )



    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs


    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

    ##################
    # Performing adversarial example testing# 
    ##################


    print '-------------------------------------'
    print 'example x :', example_x
    image = getImage(test_set[0][example_index])
    plt.figure(1)
    plt.imshow(image)

    print 'example y :', example_y
    print '-------------------------------------'

    classification, probabilities = test_example(example_index)

    if int(classification) == 0:
        print 'Correct classification performed :', True
    else:
        print 'Correct classification performed :', False
    
    print 'probabilities : ',probabilities
    print 'Number predicted :', numpy.argmax(probabilities)
    print 'with probability :',numpy.max(probabilities)*100
    print '-------------------------------------'

    gadversarial = T.vector()
    gadversarial = [T.grad(cost,x)]

    grad_cost_wrt_x = theano.function(
        inputs= [index],
        outputs=gadversarial,
        givens={
            x: test_set_x[index * 1:(index + 1) * 1],
            y: test_set_y[index * 1:(index + 1) * 1]
            }
        )

        
    print 'Creating adversarial example and trying to get results for that ...  \n \n \n '
    print '-------------------------------------'

    gradient_cost_wrt_x = grad_cost_wrt_x(example_index)

    gradient_sign =  numpy.sign(gradient_cost_wrt_x)

    gradient_sign = numpy.reshape(gradient_sign,(1,3072))

    adversarial_example_x = example_x + adversarial_parameter*gradient_sign

    input_image_x,input_image_y = shared_dataset([adversarial_example_x,example_y])

    test_input_example = theano.function(
        inputs= [],
        outputs=[classifier.errors(y),classifier.p_y_given_x],
        givens={
            x: input_image_x,
            y: input_image_y
            }
        )

    adversarial_classification, input_image_output_probilities = test_input_example()

    if int(adversarial_classification) == 0:
        print 'Correct adversarial classification performed :', True
    else:
        print 'Correct adversarial classification performed :', False
    
    image2 = getImage(adversarial_example_x)    
    plt.figure(2)
    plt.imshow(image2)
    

    print 'probabilities : ',input_image_output_probilities
    print 'Number predicted :', numpy.argmax(input_image_output_probilities)
    print 'with probability :',numpy.max(input_image_output_probilities)*100
    print '-------------------------------------'

def test_input_example(classifier,test_set_x,test_set_y,example_index = 0):
    """
    Wrapper function for testing adversarial examples
    """
    '''
    test_mlp_with_new_functionality includes the code for testing adversarial example given the example index in the dataset. 
    '''

def mlp_with_gaussian_filter(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,batch_size=128, n_hidden=500, n_hiddenLayers=3,verbose=False, smaller_set=True,std = 0.1):
    """
    Wrapper function for training and testing MLP

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient.

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization).

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization).

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer.

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type n_hidden: int or list of ints
    :param n_hidden: number of hidden units. If a list, it specifies the
    number of units in each hidden layers, and its length should equal to
    n_hiddenLayers.

    :type n_hiddenLayers: int
    :param n_hiddenLayers: number of hidden layers.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type smaller_set: boolean
    :param smaller_set: to use the smaller dataset or not to.

    """

    # load the dataset; download the dataset if it is not present
    if smaller_set:
        datasets = load_data2(ds_rate=5,std=std)
    else:
        datasets = load_data2(std=std)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    print 'n_train_batches : ',n_train_batches
    print 'n_valid_batches : ',n_valid_batches
    print 'n_test_batches : ',n_test_batches
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # TODO: construct a neural network, either MLP or CNN.
    # input, n_in, n_hidden, n_out, n_hiddenLayers
    classifier = myMLP(rng, input = x, n_in =3072 , n_hidden = n_hidden, n_out= 10, n_hiddenLayers= n_hiddenLayers)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    train_nn(train_model, validate_model, test_model,
        n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)

# test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=2,batch_size=20, n_hidden=500, n_hiddenLayers=3,verbose=True, smaller_set=False)
predict_from_trained_model3('mlp_classifier_nhidden_500_hiddenlayers_3_batchSize_128_epochs_100_json')
# predict_from_trained_model2('mlp_classifier_nhidden_500_hiddenlayers_3_batchSize_20_epochs_2')