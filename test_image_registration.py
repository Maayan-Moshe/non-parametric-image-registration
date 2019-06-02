import unittest
import numpy as np
import matplotlib.pyplot as plt

from image_registration import OpticalFlowFinder, OpticalFlowRegistrator

class TestOpticalFlowFinder(unittest.TestCase):

    def test_same_image(self):
        params = {'shape': (5, 5), 'smoothing': {'HornSchunck': 1, 'divergence': 0, 'curl': 0}}
        img = np.random.rand(*params['shape'])
        finder = OpticalFlowFinder(params=params)
        displacement = finder.solve(img, img)
        np.testing.assert_allclose(displacement['i_displacement'], np.zeros(params['shape']))
        np.testing.assert_allclose(displacement['j_displacement'], np.zeros(params['shape']))

    def test_translation(self):
        params = {'shape': (30, 20), 'smoothing': {'HornSchunck': 1, 'divergence': 0, 'curl': 0}}
        direction = np.array((1,0.5))
        origin = np.array((0.5, -0.2))
        img_0 = create_gradient_image(direction, params['shape'], np.zeros(2))
        img_1 = create_gradient_image(direction, params['shape'], origin)
        finder = OpticalFlowFinder(params=params)
        displacement = finder.solve(img_0, img_1)
        np.testing.assert_allclose(displacement['i_displacement']/displacement['j_displacement'], direction[0]/direction[1])

    def test_rotation(self):
        params = {'shape': (30, 40), 'smoothing': {'HornSchunck': 1, 'divergence': 0, 'curl': 0}}
        dir_0 = np.array((1,-1))
        dir_1 = get_rotated_direction_by_angle(dir_0, 0.5)
        origin = np.array(params['shape'])/2.
        img_0 = create_gradient_image(dir_0, params['shape'], origin)
        img_1 = create_gradient_image(dir_1, params['shape'], origin)
        diaplay_images(img_0, img_1)
        finder = OpticalFlowFinder(params=params)
        displacement = finder.solve(img_0, img_1)
        display_displacement(displacement)

    def test_gaussian_image(self):
        params = {'shape': (50, 40), 'smoothing': {'HornSchunck': 1, 'divergence': 0, 'curl': 0}}
        img_0, img_1 = get_twin_gaussian_images(params['shape'])
        diaplay_images(img_0, img_1)
        finder = OpticalFlowFinder(params)
        displacement = finder.solve(img_0, img_1)
        display_displacement(displacement)

    def test_gausian_registration(self):

        params = {'shape': (50, 40), 'smoothing': {'HornSchunck': 0.1, 'divergence': 0, 'curl': 0}, 'interpolation_degree': 1}
        img_0, img_1 = get_twin_gaussian_images(params['shape'])
        diaplay_images(img_0, img_1)
        registrator = OpticalFlowRegistrator(params= params)
        new_image = registrator.register_second_to_first(img_0, img_1)
        diaplay_images(new_image, img_0, 'new image 1', 'image_0')
        plt.show()
        pre_diff = np.average(np.abs(img_1 - img_0))
        post_diff = np.average(np.abs(new_image - img_0))
        self.assertLess(post_diff, 0.3 * pre_diff)

def get_twin_gaussian_images(shape):
    origin_av = np.array(shape) / 2.
    origin_delta = np.array(shape) / 12.
    sigma = np.array(shape) / 3.
    img_0 = create_gaussian_image(origin_av - origin_delta, sigma, shape)
    img_1 = create_gaussian_image(origin_av + origin_delta, sigma, shape)
    return img_0, img_1

def diaplay_images(img_0, img_1, title_0 = 'image_0', title_1 = 'image_1'):
    plt.figure(title_0)
    plt.imshow(img_0, 'gray', interpolation='none', origin='lower')
    plt.figure(title_1)
    plt.imshow(img_1, 'gray', interpolation='none', origin='lower')
    plt.figure(title_0 + ' - ' + title_1)
    plt.imshow(img_1 - img_0, 'gray', interpolation='none', origin='lower')

def display_displacement(displacement):
    plt.figure('displacement')
    plt.quiver(displacement['i_displacement'], displacement['j_displacement'])
    plt.show()

def get_rotated_direction_by_angle(direction, angle):

    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    rotated_dir = np.dot(R, direction)
    return rotated_dir

def create_gradient_image(direction, shape, origin):

    grid = np.moveaxis(np.indices(shape).astype(float), 0, -1)
    grid -= origin.reshape((1,1,2))
    intensity = np.dot(grid, direction)
    return intensity

def create_gaussian_image(origin, sigma, shape):
    grid = np.indices(shape).astype(float)
    grid -= origin.reshape((2,1,1))
    grid /= sigma.reshape((2,1,1))
    intensity = np.exp(-np.sum(grid*grid, axis = 0))
    return intensity