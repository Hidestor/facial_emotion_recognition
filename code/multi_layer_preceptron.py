"""
This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
import json
import numpy
import theano
import theano.tensor as T
import time

from utils import shared_dataset, load_data , getImage
from nn import LogisticRegression, HiddenLayer, myMLP, LeNetConvPoolLayer, train_nn

from theano.tensor.shared_randomstreams import RandomStreams
emotionDictionary = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}



def predict_webcam_image(classifier,x,y,timegap = .1):
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

    f = open('%s_json.save'%filename,'wb') 
    parameters = []

    for parameter in classifier.params:
        parameters.append(parameter.eval().tolist())

    parameters = parameters + [n_hidden,n_hiddenLayers]

    encode = json.dumps({'p':parameters})
    f.write(encode)
    f.close()

def load_model(filename):
    f = open(filename, 'rb') 
    filecontent= f.read()
    loaded_obj = json.loads(filecontent)
    f.close()
    print 'loaded'
    return loaded_obj['p']

def predict_from_trained_model(filename):
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

    import os
    modelFolderName = 'mlp_models'
    cmd = 'mkdir %s'%modelFolderName
    os.system(cmd)
    save_model(classifier,n_hidden,n_hiddenLayers,modelFolderName + '/'+'mlp_classifier_nhidden_%s_hiddenlayers_%s_batchSize_%s_epochs_%s'%(n_hidden,n_hiddenLayers,batch_size,n_epochs))

    print 'Model Saved. '

# modelFolderName = 'mlp_models'
# modelName = 'mlp_classifier_nhidden_500_hiddenlayers_3_batchSize_20_epochs_2_json.save'
# test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=2,batch_size=20, n_hidden=500, n_hiddenLayers=3,verbose=True, smaller_set=False)
# predict_from_trained_model(modelFolderName +'/'+modelName)