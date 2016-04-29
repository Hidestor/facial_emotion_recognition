import numpy
import cv2
import time


def get_input_image(timegap = 0.1):
	cap = cv2.VideoCapture(0)

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # Our operations on the frame come here
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    res = cv2.resize(gray,(48,48), interpolation = cv2.INTER_CUBIC)

	    # Display the resulting frame
	    cv2.imshow('frame',gray)
	    cv2.imshow('resized',res)

	    input_image = numpy.reshape(res,(48*48,1))
	    print input_image.shape

	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	    time.sleep(timegap)
	# When everything done, release the capture
	
	cap.release()
	cv2.destroyAllWindows()


get_input_image()