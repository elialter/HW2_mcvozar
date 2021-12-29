from network import *
import multiprocessing
import my_queue
import preprocessor
import os


class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        num_workers = (int)(os.environ['SLURM_CPUS_PER_TASK'])
        jobs = multiprocessing.JoinableQueue()
        self.results = my_queue.MyQueue()
        workers = []
        for i in range(num_workers):
            worker = preprocessor.Worker(jobs=jobs, result=self.results,
                                         training_data=training_data, batch_size=self.mini_batch_size)
            workers.append(worker)
            worker.start()

        # 2. Set jobs
        for i in range(self.number_of_batches * self.epochs):
            jobs.put('work')
        for j in workers:
            jobs.put('stop')

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        # 3. Stop Workers
        jobs.join()

    #        raise NotImplementedError("To be implemented")

    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        batches = []
        for i in range(self.number_of_batches):
            temp = self.results.get()
            batches.append(temp)
        return batches
#        raise NotImplementedError("To be implemented")




