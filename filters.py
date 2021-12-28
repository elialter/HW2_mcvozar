from numba import cuda, prange, float64
from numba import njit
import imageio
import matplotlib.pyplot as plt
import os
import numby as np
from numba import float64


@cuda.jit
def matmul_kernel(image, kernel, needed_result):
    x = cuda.threadIdx.x

    # the number of rows and cols in half kernel
    kernel_max_rows = kernel.shape[0] // 2
    kernel_max_cols = kernel.shape[1] // 2

    for pixel in range(x, image.shape[0] * image.shape[1], 1024):
        image_current_row = pixel // image.shape[1]
        image_current_col = pixel % image.shape[1]

        # defining where rows and cols start and end in the kernel in the due to current cell place
        starting_row = 0
        last_row = kernel.shape[0]
        start_column = 0
        last_column = kernel.shape[1]
        # check if extension from bounds exists for rows
        if image_current_row - kernel_max_rows < 0:
            starting_row = kernel_max_rows - image_current_row
        if image_current_row + kernel_max_rows >= image.shape[0]:
            last_row = kernel.shape[0] - (image_current_row + kernel_max_rows - (image.shape[0] - 1))
        # check if extension from bounds exists for cols
        if image_current_col - kernel_max_cols < 0:
            start_column = kernel_max_cols - image_current_col
        if image_current_col + kernel_max_cols >= image.shape[1]:
            last_column = kernel.shape[1] - (image_current_col + kernel_max_cols - (image.shape[1] - 1))

        # evaluate the correlation of the current pixel
        pixel_correlation = 0.0
        for i in prange(starting_row, last_row):
            for j in pragne(start_column, last_column):
                original_image_pixel = image[image_current_row - kernel_max_rows + i, col - kernel_max_cols + j]
                pixel_correlation += original_image_pixel * kernel[i, j]

        # save the correlation in the returned result array
        cuda.atomic.add(needed_result[image_current_row], image_current_col, pixel_correlation)
    cuda.syncthreads()


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    correlated_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)

    # defining our gpu elements
    gpu_image = cuda.to_device(image)
    gpu_kernel = cuda.to_device(kernel)
    gpu_correlated_image = cuda.to_device(correlated_image)

    # getting the correlation needed using gpu
    image_correlation_kernel[1, 1024](gpu_image, gpu_kernel, gpu_correlated_image)

    return gpu_correlated_image.copy_to_host(correlated_image)


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    correlated_image = np.empty_like(image, dtype=float64)

    # the number of rows and cols in half kernel
    kernel_max_rows = kernel.shape[0] // 2
    kernel_max_cols = kernel.shape[1] // 2

    for image_row in range(image.shape[0]):
        for image_col in range(image.shape[1]):
            # defining where rows and cols start and end in the kernel in the due to current cell place
            starting_row = 0
            last_row = kernel.shape[0]
            start_column = 0
            last_column = kernel.shape[1]

            # check if extension from bounds exists for rows
            if kernel_max_rows - image_row > 0:
                starting_row = kernel_max_rows - image_row
            if image_row + kernel_max_rows >= image.shape[0]:
                last_row = kernel.shape[0] - (image_row + kernel_max_rows - (image.shape[0] - 1))
            if kernel_max_cols - image_col > 0:
                start_column = kernel_max_cols - image_col
            if image_col + kernel_max_cols >= image.shape[1]:
                last_column = kernel.shape[1] - (image_col + kernel_max_cols - (image.shape[1] - 1))

            pixel_correlation = 0.0
            for i in prange(starting_row, last_row):
                for j in prange(start_column, last_column):
                    original_image_pixel = image[image_row - kernel_max_rows + i, image_col - kernel_max_cols + j]
                    pixel_correlation += original_image_pixel * kernel[i, j]
            correlated_image[image_row, image_col] = pixel_correlation

    return correlated_image


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations
    filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # filter = np.array([[1, 0, 0, 0, -1], [2, 0, 0, 0, -2], [1, 0, 0, 0, -1], [2, 0, 0, 0, -2], [1, 0, 0, 0, -1]])
    # filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1], [2, 0, -2], [1, 0, -1]])
    G_x = correlation_numba(filter, pic)
    G_y = correlation_numba(np.transpose(filter), pic)
    image_sobel = np.empty_like(pic)
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            image_sobel[i, j] = np.sqrt(np.power(G_x[i, j], 2) + np.power(G_y[i, j], 2))
    return image_sobel


def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
