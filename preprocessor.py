import multiprocessing
import numpy as np
import random
from scipy.ndimage import rotate

max_noise = 0.2

class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
#        raise NotImplementedError("To be implemented")

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        processed_image = np.reshape(image, (28, 28))
        rotate(processed_image, angle, (1, 0), False)
        return np.reshape(processed_image, 784)
#        raise NotImplementedError("To be implemented")

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        shifted_image = np.reshape(image, (28, 28))

        dy *= -1
        dx *= -1
        # shifted.show()
        shifted_image = np.roll(shifted_image, dy, axis=0)
        shifted_image = np.roll(shifted_image, dx, axis=1)
        if dx > 0:
            shifted_image[:, :dx] = 0
        elif dx < 0:
            shifted_image[:, dx:] = 0
        if dy > 0:
            shifted_image[:dy, :] = 0
        elif dy < 0:
            shifted_image[dy:, :] = 0

        return np.reshape(shifted_image, 784)
#        raise NotImplementedError("To be implemented")
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        raise NotImplementedError("To be implemented")

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        n_image = np.reshape(image,(28,28))
        n_pixels = np.empty_like(n_image)
        for i in range(n_image.shape[0]):
            for j in range(n_image.shape[1]):
                if np.floor(j + i * tilt) < 0 or np.floor(j + i * tilt) > n_image.shape[1] - 1:
                    n_pixels[i, j] = 0
                else:
                    temp = (int)(np.floor(j + i * tilt))
                    n_pixels[i, j] = np.copy(n_image[i, temp])
        return np.reshape(n_pixels, 784)
#        raise NotImplementedError("To be implemented")

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        processed_image = np.copy(image)

        for i in range(0,4):
            augement_function = random.randint(0, 3)
            if augement_function == 0:
                processed_image = Worker.add_noise(processed_image,
                                                   random.uniform(-max_noise, max_noise))
            if augement_function == 1:
                processed_image = Worker.shift(processed_image,
                                               random.randint(-2, 2), random.randint(-2, 2))
            if augement_function == 2:
                processed_image = Worker.skew(processed_image, random.uniform(-0.1, 0.1))
            if augement_function == 3:
                processed_image = Worker.rotate(processed_image, random.randint(-2, 2))

        return processed_image

#        raise NotImplementedError("To be implemented")

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            job = self.jobs.get()
            if job == 'stop':
                self.jobs.task_done()
                break
            indexes = random.sample(range(0, len(self.training_data[0])), self.batch_size)
            images=[self.process_image(self.training_data[0][i]) for i in indexes]
            t_t = np.array(images), self.training_data[1][indexes]
            self.result.put(t_t)
            self.jobs.task_done()
#        raise NotImplementedError("To be implemented")
