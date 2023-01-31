import os
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt

GRAYSCALE = 1
RGB = 2
DOWN_SIZE = 2
UP_SIZE = 2
MIN_IMAGE_SIZE = 16
EXAMPLE_2_IMAGE_1 = 'externals/rafael-nadal.jpg'
EXAMPLE_2_IMAGE_2 ='externals/roger-federer.jpg'
EXAMPLE_2_MOCK = 'externals/rafael-nadal-mock.jpg'
EXAMPLE_1_IMAGE_1 = 'externals/surf.jpg'
EXAMPLE_1_IMAGE_2 ='externals/desierto.jpg'
EXAMPLE_1_MOCK = 'externals/surf-mock.jpg'


def normalize(im, normalizer=255):
    return im / normalizer


def normalize_256bin(im, normalizer=255):
    return im * normalizer


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # If the input image is grayscale, we won’t call it with representation = 2.
    if representation == GRAYSCALE:
        return rgb2gray(normalize(imread(filename))).astype(np.float64)
    return normalize(imread(filename)).astype(np.float64)


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    return helper_down_sample(helper_blur_filter(im, blur_filter))


def helper_blur_filter(im, blur_filter):
    return convolve(convolve(im, blur_filter), np.transpose(blur_filter))


def helper_adjust_image_size(im1, im2):
    """
    Given two 2D np.arrays adjusts their dimensions to be equals by picking the minimum dimension
    :param im1:
    :param im2:
    :return:
    """
    if im1.shape[0] < im2.shape[0]:
        im2 = im2[:im1.shape[0], :im2.shape[1]]
    if im2.shape[0] < im1.shape[0]:
        im1 = im2[:im2.shape[0], :im1.shape[1]]
    if im1.shape[1] < im2.shape[1]:
        im2 = im2[:im2.shape[0], :im1.shape[1]]
    if im2.shape[1] < im1.shape[1]:
        im1 = im2[:im1.shape[0], :im2.shape[1]]
    return im1, im2


def helper_down_sample(im, scale=DOWN_SIZE):
    return im[::scale, ::scale]


def helper_set_zero_size(im, scale):
    return tuple(scale * np.array(im.shape))


def helper_expand_sample(im, scale=UP_SIZE):
    size = helper_set_zero_size(im, scale)
    zeros_im = np.zeros(size)
    zeros_im[::scale, ::scale] = im
    return zeros_im


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    return helper_blur_filter(helper_expand_sample(im), blur_filter * UP_SIZE)


def helper_get_convolution_filter(filter_size):
    curr_filter = np.array([1, 1])
    for k in range(2, filter_size):
        curr_filter = np.convolve(curr_filter, [1, 1])
    return np.array([normalize(curr_filter, np.cumsum(curr_filter)[-1])])


def helper_handle_gaussian_reduce(curr_g, max_levels, filter_vec, pyr):
    if len(pyr) == max_levels or curr_g.shape[0] < MIN_IMAGE_SIZE or curr_g.shape[1] < MIN_IMAGE_SIZE:
        return
    pyr.append(curr_g)
    helper_handle_gaussian_reduce(reduce(curr_g, filter_vec), max_levels, filter_vec, pyr)


def helper_handle_laplacian_reduce(curr_g, max_levels, filter_vec, pyr):
    if len(pyr) == max_levels - 1 or (curr_g.shape[0] < MIN_IMAGE_SIZE or curr_g.shape[1] < MIN_IMAGE_SIZE):
        pyr.append(curr_g)
        return
    reduced_g = reduce(curr_g, filter_vec)
    # adjusts the size, in case there dimensions are not equal, we get rid of last row or column.
    curr_g, im_expand = helper_adjust_image_size(curr_g, expand(reduced_g, filter_vec))
    pyr.append(curr_g - im_expand)
    helper_handle_laplacian_reduce(reduced_g, max_levels, filter_vec, pyr)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    if filter_size < 2:
        return
    filter_vec, pyr = helper_get_convolution_filter(filter_size), []
    helper_handle_gaussian_reduce(im, max_levels, filter_vec, pyr)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    if filter_size < 2:
        return
    filter_vec, pyr = helper_get_convolution_filter(filter_size), []
    helper_handle_laplacian_reduce(im, max_levels, filter_vec, pyr)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    filtered_lpyr = [coeff[i] * lpyr[i] for i in range(len(coeff))]
    g_im = filtered_lpyr[-1] * coeff[-1]
    for i in range(len(coeff) - 2, -1, -1):
        g_im = filtered_lpyr[i] + expand(g_im, filter_vec)
    return g_im


def helper_stretch_image(im):
    return (im - np.amin(im)) / (np.amax(im) - np.amin(im))


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
    from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    curr_im = helper_stretch_image(pyr[0])
    x_shape = curr_im.shape[0]
    for k in range(1, levels):
        pad_width = ((0, x_shape - pyr[k].shape[0]), (0, 0))
        to_pad = np.pad(helper_stretch_image(pyr[k]), pad_width, mode="constant")
        curr_im = np.concatenate((curr_im, to_pad), 1)
    return curr_im


def display_pyramid(pyr, levels):
    """
    display the rendered pyramid
    :param pyr:
    :param levels:
    """
    plt.imshow(render_pyramid(pyr, levels), cmap=plt.cm.gray)
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    lap_1, filter_vec_lap_1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_2, filter_vec_lap_2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gauss_m, filter_vec_gauss_m = build_gaussian_pyramid(mask.astype('float64'), max_levels, filter_size_mask)
    lap_out = [gauss_m[k] * lap_1[k] + (1 - gauss_m[k]) * lap_2[k] for k in range(len(lap_2))]
    return np.clip(laplacian_to_image(lap_out, filter_vec_lap_1, [1] * len(lap_2)), 0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blend_pyramid_rgb(im1, im2, mask, blended_im):
    blended_im[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 3, 3, 3)
    blended_im[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 3, 3, 3)
    blended_im[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 3, 3, 3)


def display_blending(blended_im, im1, im2, mask):
    fig, ((blend1, blend2), (blend3, blend4)) = plt.subplots(2, 2)
    blend1.imshow(im1)
    blend2.imshow(im2)
    blend3.imshow(mask, cmap=plt.cm.gray)
    blend4.imshow(blended_im)
    plt.show()
    plt.figure()
    plt.imshow(blended_im)
    plt.show()


def get_images_to_blend(path1, path2, path3):
    return read_image(relpath(path1), 2), read_image(relpath(path2), 2), np.array(imread(relpath(path3)), dtype="bool")


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1, im2, mask = get_images_to_blend(EXAMPLE_1_IMAGE_1, EXAMPLE_1_IMAGE_2, EXAMPLE_1_MOCK)
    blended_im = np.zeros(np.shape(im1))
    blend_pyramid_rgb(im1, im2, mask, blended_im)
    display_blending(blended_im, im1, im2, mask)
    return im1, im2, mask, blended_im


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1, im2, mask = get_images_to_blend(EXAMPLE_2_IMAGE_1, EXAMPLE_2_IMAGE_2, EXAMPLE_2_MOCK)
    blended_im = np.zeros(np.shape(im1))
    blend_pyramid_rgb(im1, im2, mask, blended_im)
    display_blending(blended_im, im1, im2, mask)
    return im1, im2, mask, blended_im


if __name__ == "__main__":
    blending_example1()
    blending_example2()
    display_pyramid(build_laplacian_pyramid(read_image("externals/monkey.jpg", 2), 3, 5)[0], 3)
