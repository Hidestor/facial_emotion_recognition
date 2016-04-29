"""
This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from utils import shared_dataset, load_data
from nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn
emotionDictionary = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
import time

def save_model(filename,params,learning_rate, n_epochs, nkerns,batch_size, verbose,filterwidth_layer0,filterheight_layer0,poolsize_layer0,filterwidth_layer1,filterheight_layer1,poolsize_layer1,filterwidth_layer2,filterheight_layer2,poolsize_layer2,neurons_hidden ,smaller_set):
    f = open('%s_json.save'%filename,'wb') 
    parameters = []
    import json

    for parameter in params:
        parameters.append(parameter.eval().tolist())


    encode = json.dumps({'p':parameters,
        'learning_rate':learning_rate,
        'n_epochs':n_epochs, 
        'nkerns':nkerns,
        'batch_size':batch_size, 
        'verbose':verbose,
        'filterwidth_layer0':filterwidth_layer0,'filterheight_layer0':filterheight_layer0,'poolsize_layer0':poolsize_layer0,'filterwidth_layer1':filterwidth_layer1,'filterheight_layer1':filterheight_layer1,'poolsize_layer1':poolsize_layer1,'filterwidth_layer2':filterwidth_layer2,'filterheight_layer2':filterheight_layer2,'poolsize_layer2':poolsize_layer2,'neurons_hidden':neurons_hidden ,'smaller_set':smaller_set})
    f.write(encode)
    f.close()
    
def load_model(filename):
    import json
    f = open('%s_json.save'%filename, 'rb') 
    filecontent= f.read()
    loaded_obj = json.loads(filecontent)
    f.close()
    print 'loaded'

    parameters_shared = loaded_obj['p']
    learning_rate = loaded_obj['learning_rate']
    n_epochs= loaded_obj['n_epochs'] 
    nkerns= loaded_obj['nkerns']
    batch_size= 1 
    verbose= loaded_obj['verbose']
    filterwidth_layer0= loaded_obj['filterwidth_layer0']
    filterheight_layer0= loaded_obj['filterheight_layer0']
    poolsize_layer0= loaded_obj['poolsize_layer0']
    filterwidth_layer1= loaded_obj['filterwidth_layer1']
    filterheight_layer1= loaded_obj['filterheight_layer1']
    poolsize_layer1= loaded_obj['poolsize_layer1']
    filterwidth_layer2= loaded_obj['filterwidth_layer2']
    filterheight_layer2= loaded_obj['filterheight_layer2']
    poolsize_layer2= loaded_obj['poolsize_layer2']
    neurons_hidden = loaded_obj['neurons_hidden']
    smaller_set= loaded_obj['smaller_set']


    parameters = []

    for p in parameters_shared:
        p1 = []
        for row in p:
            p1.append(numpy.asarray(row))
        parameters.append(numpy.asarray(p1))

    rng = numpy.random.RandomState(23455)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))


    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape= (batch_size, 1, 48, 48),
        filter_shape= (nkerns[0],1,filterwidth_layer0,filterheight_layer0),
        W= parameters[-2],
        b=parameters[-1],
        poolsize= (poolsize_layer0,poolsize_layer0)
    )
    
    print '-------------------------------------------------------------------------------------------- \n'
    layer0_outputwidth,layer0_outputheight = ( (48-filterwidth_layer0+1)/poolsize_layer0,(48-filterheight_layer0+1)/poolsize_layer0 )
    print 'Layer0 build. Shape of feature map  :',layer0_outputwidth, layer0_outputheight, 'Number of feature maps : ',nkerns[0]
    
    print '-------------------------------------------------------------------------------------------- \n'
    
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],layer0_outputwidth,layer0_outputheight),
        filter_shape= (nkerns[1],nkerns[0],filterwidth_layer1,filterheight_layer1),
        W= parameters[-4],
        b=parameters[-3],
        poolsize=(poolsize_layer1,poolsize_layer1)
 
    )

    layer1_outputwidth,layer1_outputheight = (layer0_outputwidth-filterwidth_layer1+1)/poolsize_layer1,(layer0_outputheight-filterwidth_layer1+1)/poolsize_layer1  
    print 'Layer1 build. Shape of feature map :',layer1_outputwidth,layer1_outputheight, 'Number of feature maps : ',nkerns[1]
    
    print '-------------------------------------------------------------------------------------------- \n'
    poolsize_width_layer0_to_layer1 = layer0_outputwidth/layer1_outputwidth
    poolsize_height_layer0_to_layer1 = layer0_outputheight/layer1_outputheight
    print 'poolsize layer 0 o/p to layer 1 o/p width :',layer0_outputwidth/layer1_outputwidth
    print 'poolsize layer 0 o/p to layer 1 o/p height :',layer0_outputheight/layer1_outputheight
    
    
    layer0_output_ds = downsample.max_pool_2d(
            input=layer0.output,
            ds=(poolsize_width_layer0_to_layer1,poolsize_height_layer0_to_layer1), # TDOD: change ds
            ignore_border=True
    )

    # concatenate layer
    print 'max pool layer created. between output of layer0 and output of layer1. output of this max pool layer : ',layer0_outputwidth/poolsize_width_layer0_to_layer1,layer0_outputheight/poolsize_height_layer0_to_layer1
    print '-------------------------------------------------------------------------------------------- \n'
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)

    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape= (batch_size,nkerns[0]+nkerns[1],layer1_outputwidth,layer1_outputheight),
        filter_shape= (nkerns[2],nkerns[0]+nkerns[1],filterwidth_layer2,filterheight_layer2),
        W= parameters[-6],
        b=parameters[-5],
        poolsize=(poolsize_layer2,poolsize_layer2)        
    )
    
    print 'Input to Layer2 (not equal to output of Layer1) : ', nkerns[0]+nkerns[1]
    layer2_outputwidth,layer2_outputheight = (layer1_outputwidth-filterwidth_layer2+1)/poolsize_layer2,(layer1_outputheight-filterwidth_layer2+1)/poolsize_layer2  
    print 'Layer2 build. Shape of feature map :',layer2_outputwidth,layer2_outputheight, 'Number of feature maps : ',nkerns[2]

    print '-------------------------------------------------------------------------------------------- \n'
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * layer2_outputwidth * layer2_outputwidth,
        n_out= neurons_hidden,
        W= parameters[-8],
        b=parameters[-7],
        activation=T.tanh
    )
    
    print 'MLP Layer created. Input neurons : ',nkerns[2] * layer2_outputwidth * layer2_outputwidth, ' Output neurons :',neurons_hidden
    print '-------------------------------------------------------------------------------------------- \n'
    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
        n_in= neurons_hidden,
        n_out=7,
        W= parameters[-10],
        b=parameters[-9])

    print 'Logistic Layer created. Input neurons : ',neurons_hidden, ' output neurons :',10
  
    print '-------------------------------------------------------------------------------------------- \n'
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    
 
    print 'Model Created...'
    ###############
    # MAKE PREDICTION #
    ###############
    import cv2
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
        batch_size = 1
        #Close the frame when 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        


        res = res/255
        temp = numpy.asarray([1])
        input_image_x,input_image_y = shared_dataset([input_image,temp])

        test_input_example = theano.function(
            inputs= [],
            outputs=[layer4.errors(y),layer4.p_y_given_x],
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
        
        time.sleep(0.5)

def test_lenet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],batch_size=200, verbose=False,filterwidth_layer0=2,filterheight_layer0=2,poolsize_layer0=2,filterwidth_layer1 = 2,filterheight_layer1=2,poolsize_layer1 = 1,neurons_layer2 = 500,smaller_set= False):
    """
    Wrapper function for testing LeNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)
    
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

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

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 48, 48),
        filter_shape=(nkerns[0],1,filterwidth_layer0,filterheight_layer0),
        poolsize=(poolsize_layer0,poolsize_layer0)
    )

    # At the output of convo layer in the layer0 the output size reduces to 32 - filterwidth + 1,32 - filterheight + 1
    # At output of the (32-filterwidth+1/poolsize,(32-filterwidth+1)/poolsize)
    
    # TODO: Construct the second convolutional pooling layer
    layer0_outputwidth,layer0_outputheight = ( (48-filterwidth_layer0+1)/poolsize_layer0,(48-filterheight_layer0+1)/poolsize_layer0 )
    print '-------------------------------------------------------------------------------------------- \n'

    print 'Layer0 build. Shape of feature map  :',layer0_outputwidth,layer0_outputheight, 'Number of feature maps : ',nkerns[0]

    
    layer1 = LeNetConvPoolLayer(
        rng,
        input= layer0.output,
        image_shape= (batch_size,nkerns[0],layer0_outputwidth,layer0_outputheight),
        filter_shape= (nkerns[1],nkerns[0],filterwidth_layer1,filterheight_layer1),
        poolsize=(poolsize_layer1,poolsize_layer1)
    )
        
    print '-------------------------------------------------------------------------------------------- \n'
    layer1_outputwidth,layer1_outputheight = (layer0_outputwidth-filterwidth_layer1+1)/poolsize_layer1,(layer0_outputheight-filterwidth_layer1+1)/poolsize_layer1  
    print 'Layer1 build. Shape of feature map :',layer1_outputwidth,layer1_outputheight, 'Number of feature maps : ',nkerns[1]

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    
    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input= layer2_input,
        n_in= nkerns[1]*layer1_outputwidth*layer1_outputheight,
        n_out= neurons_layer2,
        activation= T.tanh
    )
    print '-------------------------------------------------------------------------------------------- \n'

    print 'Layer2 build - MLP layer. Input neurons : ',nkerns[1]*layer1_outputwidth*layer1_outputheight, ' output neurons : ',neurons_layer2
    
    
    # TODO: classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
         input=layer2.output,
         n_in= neurons_layer2,
    n_out=7)

    print '-------------------------------------------------------------------------------------------- \n'
    print 'Logistic Regression layer build. Input neurons: ',neurons_layer2,' Output neurons :',7
    
    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params =layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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

def test_convnet(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512, 20],batch_size=200, verbose=False,filterwidth_layer0=2,filterheight_layer0=2,poolsize_layer0=2,filterwidth_layer1=2,filterheight_layer1=2,poolsize_layer1=1,filterwidth_layer2=2,filterheight_layer2=2,poolsize_layer2=1,neurons_hidden = 300,smaller_set= False):
    """
    Wrapper function for testing Multi-Stage ConvNet on SVHN dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_szie: number of examples in minibatch.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    """

    rng = numpy.random.RandomState(23455)

    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

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

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))


    # TODO: Construct the first convolutional pooling layer:
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape= (batch_size, 1, 48, 48),
        filter_shape= (nkerns[0],1,filterwidth_layer0,filterheight_layer0),
        poolsize= (poolsize_layer0,poolsize_layer0)
    )
    
    print '-------------------------------------------------------------------------------------------- \n'
    layer0_outputwidth,layer0_outputheight = ( (48-filterwidth_layer0+1)/poolsize_layer0,(48-filterheight_layer0+1)/poolsize_layer0 )
    print 'Layer0 build. Shape of feature map  :',layer0_outputwidth, layer0_outputheight, 'Number of feature maps : ',nkerns[0]
    
    print '-------------------------------------------------------------------------------------------- \n'
    
    

    # TODO: Construct the second convolutional pooling layer
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size,nkerns[0],layer0_outputwidth,layer0_outputheight),
        filter_shape= (nkerns[1],nkerns[0],filterwidth_layer1,filterheight_layer1),
        poolsize=(poolsize_layer1,poolsize_layer1)
 
    )

    layer1_outputwidth,layer1_outputheight = (layer0_outputwidth-filterwidth_layer1+1)/poolsize_layer1,(layer0_outputheight-filterwidth_layer1+1)/poolsize_layer1  
    print 'Layer1 build. Shape of feature map :',layer1_outputwidth,layer1_outputheight, 'Number of feature maps : ',nkerns[1]
    #
    # Combine Layer 0 output and Layer 1 output
    # TODO: downsample the first layer output to match the size of the second
    # layer output.
    
    print '-------------------------------------------------------------------------------------------- \n'
    poolsize_width_layer0_to_layer1 = layer0_outputwidth/layer1_outputwidth
    poolsize_height_layer0_to_layer1 = layer0_outputheight/layer1_outputheight
    print 'poolsize layer 0 o/p to layer 1 o/p width :',layer0_outputwidth/layer1_outputwidth
    print 'poolsize layer 0 o/p to layer 1 o/p height :',layer0_outputheight/layer1_outputheight
    
    
       
    layer0_output_ds = downsample.max_pool_2d(
            input=layer0.output,
            ds=(poolsize_width_layer0_to_layer1,poolsize_height_layer0_to_layer1), # TDOD: change ds
            ignore_border=True
    )
    # concatenate layer
    print 'max pool layer created. between output of layer0 and output of layer1. output of this max pool layer : ',layer0_outputwidth/poolsize_width_layer0_to_layer1,layer0_outputheight/poolsize_height_layer0_to_layer1
    print '-------------------------------------------------------------------------------------------- \n'
    layer2_input = T.concatenate([layer1.output, layer0_output_ds], axis=1)

    # TODO: Construct the third convolutional pooling layer
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer2_input,
        image_shape= (batch_size,nkerns[0]+nkerns[1],layer1_outputwidth,layer1_outputheight),
        filter_shape= (nkerns[2],nkerns[0]+nkerns[1],filterwidth_layer2,filterheight_layer2),
        poolsize=(poolsize_layer2,poolsize_layer2)        
    )
    
    print 'Input to Layer2 (not equal to output of Layer1) : ', nkerns[0]+nkerns[1]
    layer2_outputwidth,layer2_outputheight = (layer1_outputwidth-filterwidth_layer2+1)/poolsize_layer2,(layer1_outputheight-filterwidth_layer2+1)/poolsize_layer2  
    print 'Layer2 build. Shape of feature map :',layer2_outputwidth,layer2_outputheight, 'Number of feature maps : ',nkerns[2]

    print '-------------------------------------------------------------------------------------------- \n'
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 1 * 1).
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * layer2_outputwidth * layer2_outputwidth,
        n_out= neurons_hidden,
        activation=T.tanh
    )
    
    print 'MLP Layer created. Input neurons : ',nkerns[2] * layer2_outputwidth * layer2_outputwidth, ' Output neurons :',neurons_hidden
    print '-------------------------------------------------------------------------------------------- \n'
    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output,
        n_in= neurons_hidden,
        n_out=7)

    print 'Logistic Layer created. Input neurons : ',neurons_hidden, ' output neurons :',10
  
    print '-------------------------------------------------------------------------------------------- \n'
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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

    train_nn(train_model, validate_model, test_model,n_train_batches, n_valid_batches, n_test_batches, n_epochs, verbose)
    
    ###############
    # SAVE MODEL #
    ###############

    save_model('convo_model',params,learning_rate, n_epochs, nkerns,batch_size, verbose,filterwidth_layer0,filterheight_layer0,poolsize_layer0,filterwidth_layer1,filterheight_layer1,poolsize_layer1,filterwidth_layer2,filterheight_layer2,poolsize_layer2,neurons_hidden ,smaller_set)
    print 'Model saved.'    

def test_CDNN(learning_rate=0.1, n_epochs=1000, nkerns=[16, 512],batch_size=200, verbose=False,filterwidth_layer0=3,filterheight_layer0=3,poolsize_layer0=2,filterwidth_layer1 = 6,filterheight_layer1=6,poolsize_layer1 = 2,neurons_layer2 = 300,neurons_layer3= 300,smaller_set= False):
    """
    Wrapper function for testing CNN in cascade with DNN
    """
    rng = numpy.random.RandomState(23455)
    
    if smaller_set:
        datasets = load_data(ds_rate=5)
    else:
        datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

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

    # Reshape matrix of rasterized images of shape (batch_size, 3 * 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 48, 48))

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 48, 48),
        filter_shape=(nkerns[0],1,filterwidth_layer0,filterheight_layer0),
        poolsize=(poolsize_layer0,poolsize_layer0)
    )

    # At the output of convo layer in the layer0 the output size reduces to 32 - filterwidth + 1,32 - filterheight + 1
    # At output of the (32-filterwidth+1/poolsize,(32-filterwidth+1)/poolsize)
    
    # TODO: Construct the second convolutional pooling layer
    layer0_outputwidth,layer0_outputheight = ( (48-filterwidth_layer0+1)/poolsize_layer0,(48-filterheight_layer0+1)/poolsize_layer0 )
    print '-------------------------------------------------------------------------------------------- \n'

    print 'Layer0 build. Shape of feature map  :',layer0_outputwidth,layer0_outputheight, 'Number of feature maps : ',nkerns[0]

    
    layer1 = LeNetConvPoolLayer(
        rng,
        input= layer0.output,
        image_shape= (batch_size,nkerns[0],layer0_outputwidth,layer0_outputheight),
        filter_shape= (nkerns[1],nkerns[0],filterwidth_layer1,filterheight_layer1),
        poolsize=(poolsize_layer1,poolsize_layer1)
    )
        
    print '-------------------------------------------------------------------------------------------- \n'
    layer1_outputwidth,layer1_outputheight = (layer0_outputwidth-filterwidth_layer1+1)/poolsize_layer1,(layer0_outputheight-filterwidth_layer1+1)/poolsize_layer1  
    print 'Layer1 build. Shape of feature map :',layer1_outputwidth,layer1_outputheight, 'Number of feature maps : ',nkerns[1]

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    layer2_input = layer1.output.flatten(2)
    
    # TODO: construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input= layer2_input,
        n_in= nkerns[1]*layer1_outputwidth*layer1_outputheight,
        n_out= neurons_layer2,
        activation= T.tanh
    )
    print '-------------------------------------------------------------------------------------------- \n'

    print 'Layer2 build - MLP layer. Input neurons : ',nkerns[1]*layer1_outputwidth*layer1_outputheight, ' output neurons : ',neurons_layer2
    
    
    
    layer3 = HiddenLayer(
        rng,
        input= layer2.output,
        n_in= neurons_layer2,
        n_out= neurons_layer3,
        activation= T.tanh
    )
    print '-------------------------------------------------------------------------------------------- \n'

    print 'Layer3 build - MLP layer. Input neurons : ',neurons_layer2, ' output neurons : ',neurons_layer3

    
    # TODO: classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(
         input=layer3.output,
         n_in= neurons_layer3,
    n_out=7)

    print '-------------------------------------------------------------------------------------------- \n'
    print 'Logistic Regression layer build. Input neurons: ',neurons_layer3,' Output neurons :',10
    
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)
    
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # TODO: create a list of all model parameters to be fit by gradient descent
    params =layer4.params +layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
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

load_model('convo_model')
# test_convnet(learning_rate=0.1, n_epochs=1, nkerns=[1, 2, 2],batch_size=1, verbose=False,filterwidth_layer0=2,filterheight_layer0=2,poolsize_layer0=2,filterwidth_layer1=1,filterheight_layer1=1,poolsize_layer1=1,filterwidth_layer2=2,filterheight_layer2=2,poolsize_layer2=1,neurons_hidden = 300,smaller_set= False)