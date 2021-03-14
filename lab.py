#!/usr/bin/env python3

import math

from PIL import Image as Image

# NO ADDITIONAL IMPORTS ALLOWED!

def get_pixel(image, x, y):
    # handles out of bound coordinates
    if x < 0: 
        x = 0
    elif x >= image['height']:
        x = image['height'] - 1
    if y < 0: 
        y = 0
    elif y >= image['width']:
        y = image['width'] - 1
    # used helper function to get index of (x, y) pixel in 1D array
    return image['pixels'][get_pixel_index(image, x, y)]
    #return image['pixels'][x, y)


def set_pixel(image, x, y, c):
    # used helper function to get index of (x, y) pixel in 1D array
    image['pixels'][get_pixel_index(image, x, y)] = c
    #image['pixels'][x, y] = c


def apply_per_pixel(image, func):
    result = {
        'height': image['height'],
        'width': image['width'], # fixed typo in width (widht -> width)
        #'widht': image['width'],
        'pixels': image['pixels'].copy() # initialized results with pixels from original image
        #'pixels': [],
    }
    for x in range(image['height']):
        for y in range(image['width']):
            color = get_pixel(image, x, y)
            newcolor = func(color)
            # moved set pixel call inside of the second for loop, so it runs for each pixel and fixed the order of params, x and y
            set_pixel(result, x, y, newcolor)
        #set_pixel(result, y, x, newcolor)
    return result


def inverted(image):
    # changed 256-c to 255-c
    return apply_per_pixel(image, lambda c: 255-c)

# added a helper function to return pixel index in flat array, to fix "trying to index into array with a tuple"
def get_pixel_index(image, x, y):
    """
    Calculates and returns the index of the (x, y) pixel in the 1D pixels array
    """
    return (image['width'] * x) + y

# HELPER FUNCTIONS

def correlate(image, kernel):
    """
    Compute the result of correlating the given image with the given kernel.

    The output of this function should have the same form as a 6.009 image (a
    dictionary with 'height', 'width', and 'pixels' keys), but its pixel values
    do not necessarily need to be in the range [0,255], nor do they need to be
    integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE
    kernel - 2D float array
        example 3x3 kernel - 
            [[1/9 1/9 1/9],
            [1/9 1/9 1/9],
            [1/9 1/9 1/9]]
    """
    correlated_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }

    kernel_size = len(kernel)
    # loops over the coords in the image
    for x in range(image['height']):
        for y in range(image['width']):
            # for each coord it loops over the kernel and calculates a correlated sum
            correlated_sum = 0
            # initializes image_x to the x pos in the image that corresponds to row 0 in the kernel
            image_x = int(x - ((kernel_size - 1) / 2))
            for i in range(kernel_size):
                # initializes image_y to the y pos in the image that corresponds to col 0 in the kernel
                image_y = int(y - ((kernel_size - 1) / 2))
                for j in range(kernel_size):
                    # get_pixel handles if the coords are out of bounds
                    correlated_sum += get_pixel(image, image_x, image_y) * kernel[i][j]
                    image_y += 1
                image_x += 1
            # updates the pixels value to be the new sum
            set_pixel(correlated_image, x, y, correlated_sum)
    return correlated_image

def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the 'pixels' list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    for x in range(image['height']):
        for y in range(image['width']):
            value = round(get_pixel(image, x, y))
            if value > 255:
                value = 255
            elif value < 0:
                value = 0
            set_pixel(image, x, y, value)

def create_box_blur_kernel(n):
    """
    Takes in a value n and outputs a 2d array of size n x n whose values sum to 1
    """
    total_values = n * n
    k = [[1/total_values] * n] * n
    return k


# FILTERS

def blurred(image, n):
    """
    Return a new image representing the result of applying a box blur (with
    kernel size n) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    blurred_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    box_blur = create_box_blur_kernel(n)

    # then compute the correlation of the input image with that kernel
    blurred_image = correlate(image, box_blur)

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(blurred_image)
    return blurred_image

def sharpened(image, n):
    """
    Returns a new sharpened image (also called unsharp mask) by subtracting a 
    blurred version of the image from a scaled version of the original image

    This process does not mutate the input image; rather, it creates a separate 
    structure to represent the output.
    """
    # create a new image with the same height and width as the inputed image and double all of the pixel values
    sharpened_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': [ (2 * i) for i in image['pixels'] ]
    }
    blurred_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': image['pixels'].copy()
    }
    # create the box blur kernel
    box_blur = create_box_blur_kernel(n)
    # compute the correlation of the inputed image with the box blur
    blurred_image = correlate(image, box_blur)

    # loop over the pixels in the sharpened image and subtract the corresponding blurred image pixel
    for i in range(len(sharpened_image['pixels'])):
        sharpened_image['pixels'][i] -= blurred_image['pixels'][i]

    # make sure it is a valid image
    round_and_clip_image(sharpened_image)
    return sharpened_image

def edges(image):
    """
    Implements the Sobel operator filter, which is useful for detecting edges in images
        - Performs 2 separate correlations (one with k_x and one with k_y)
        - the resulting image is a combination of the 2 correlated images, according to the formula below
            round([c1^2+c2^2]^(1/2))
    """
    k_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    k_y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    # inputed image correlated with kernel x
    o_x = correlate(image, k_x)
    # inputed image correlated with kernel y
    o_y = correlate(image, k_y)
    # setup a new image object with the same height and width as the inputed image
    edge_detected_image = {
        'height': image['height'],
        'width': image['width'],
        'pixels': []
    }
    # for each coord, calculate the square root of the sum of both correlated images squared
    for x in range(image['height']):
        for y in range(image['width']):
            edge_detected_image['pixels'].append(round(math.sqrt(get_pixel(o_x, x, y)*get_pixel(o_x, x, y) + get_pixel(o_y, x, y)*get_pixel(o_y, x, y))))
    # make sure it is a valid image
    round_and_clip_image(edge_detected_image)
    return edge_detected_image
# HELPER FUNCTIONS FOR LOADING AND SAVING IMAGES

def load_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_image('test_images/cat.png')
    """
    with open(filename, 'rb') as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith('RGB'):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == 'LA':
            pixels = [p[0] for p in img_data]
        elif img.mode == 'L':
            pixels = list(img_data)
        else:
            raise ValueError('Unsupported image mode: %r' % img.mode)
        w, h = img.size
        return {'height': h, 'width': w, 'pixels': pixels}


def save_image(image, filename, mode='PNG'):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode='L', size=(image['width'], image['height']))
    out.putdata(image['pixels'])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.
    pass

    # Correlation 4.4 - 
    # pigbird = load_image('test_images/pigbird.png')
    # kernel = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    # save_image(correlate(pigbird, kernel), "test_images/correlated_pigbird.png")

    # Blurred 5.1 -
    # cat = load_image('test_images/cat.png')
    # save_image(blurred(cat, 5), "test_images/blurred_cat.png")

    # Sharpened 5.2 -
    # python = load_image('test_images/python.png')
    # save_image(sharpened(python, 11), "test_images/sharpened_python.png")

    # Edge Detection 5.2 -
    # construct = load_image('test_images/construct.png')
    # save_image(edges(construct), "test_images/edges_construct.png")

