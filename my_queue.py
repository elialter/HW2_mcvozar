from multiprocessing import Lock, Pipe


class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.reading_pipe, self.writing_pipe = Pipe()
        self.my_lock = Lock()

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.my_lock.acquire()
        self.writing_pipe.send(msg)
        self.my_lock.release()

    def get(self):
        '''Get the next message from queue (FIFO)

        Return
        ------
        An object
        '''
        return self.reading_pipe.recv()

