import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.array input data

     Returns
     -------
     np.array
         sigmoid of the input
     """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.array input data

         Returns
         -------
         np.array
             derivative of sigmoid of the input x
    """
    sig_prime = sigmoid(x) * (1.0 - sigmoid(x))
    return sig_prime


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of xavier initialized np arrays weight matrices
    """
    xavier_list = []
    loop_size = len(sizes) - 1
    for i in range(0, loop_size):
        xavier_list.append(np.array(xavier_initialization(sizes[i], sizes[i+1])))
    return xavier_list




def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays weight matrices
    """
    zero_list = []
    for i in range(0, len(sizes) - 1):
        zero_list.append(np.zeros([sizes[i], sizes[i+1]], dtype = int))
    return zero_list


def zeros_biases(list):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list
             list of zero np arrays bias matrices

    """
    zero_list = []
    for i in list:
        zero_list.append(np.zeros(i))

    return zero_list


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.array of input data
         labels : np.array of input labels
         batch_size : int size of batch

         Returns
         -------
         list
             list of tuples of (data batch of batch_size, labels batch of batch_size)


    """
    needed_list = []
    for i in range(0, len(data) - batch_size + 1, batch_size):
        needed_list.append((data[i : i + batch_size], labels[i : i + batch_size]))
    needed_list += [(data[len(data) - (len(data) % batch_size) : len(data)], labels[len(labels) - (len(labels) % batch_size) : len(data)])]
    return needed_list


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.array of numbers
         list2 : np.array of numbers

         Returns
         -------
         list
             list of sum of each two elements by index
    """
    sum_list = []
    for i in range(0, len(list1)):
        sum_list.append(list1[i] + list2[i])
    return sum_list


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
