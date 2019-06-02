import numpy as np
from scipy.sparse import coo_matrix

class OpticalFlowRegistrator:

    def __init__(self, params):
        self.indexes = np.indices(params['shape'])
        self.i_s = range(params['shape'][0]),
        self.j_s = range(params['shape'][1])
        self.flow_finder = OpticalFlowFinder(params)
        self.params = params

    def register_second_to_first(self, img_0, img_1):
        from scipy.interpolate import RectBivariateSpline

        displacement = self.flow_finder.solve(img_0, img_1)
        k = self.params['interpolation_degree']
        resampler = RectBivariateSpline(self.i_s, self.j_s, img_1, kx=k, ky=k)
        new_i = self.indexes[0] + displacement['i_displacement']
        new_j = self.indexes[1] + displacement['j_displacement']
        new_image = resampler.ev(new_i, new_j)
        return new_image

class OpticalFlowFinder:
    #TODO check the factor 2
    def __init__(self, params):

        self.shape = params['shape']
        self.smooth_coef = params['smoothing']
        self.length = self.shape[0]*self.shape[1]
        self.indexes = range(2*self.length)
        self.coo_hess = CoordinateHessian(self.shape)
        self.__build_smoothness_term()

    def solve(self, image_0, image_1):
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr

        flow_builder = OpticalFlowBuilder(image_0, image_1, {'shape': self.shape, 'indexes': self.indexes})
        coo_hess = flow_builder.coo_hess
        coo_hess.add(self.smooth_hess)
        #TODO check if lsqr solver is better than LU solver with some addition to the diagonal
        solution = lsqr(csr_matrix(coo_hess()), flow_builder.linear_term)[0]
        i_displacement = solution[:self.length].reshape(self.shape)
        j_displacement = solution[self.length:].reshape(self.shape)
        return {'i_displacement': i_displacement, 'j_displacement': j_displacement}

    def __build_smoothness_term(self):

        smooth_params = {'indexes': self.indexes, 'length': self.length,
                         'shape': self.shape, 'smoothing': self.smooth_coef}

        builder = SmoothingBuilder(smooth_params)
        self.smooth_hess = builder.coo_hess

class OpticalFlowBuilder:

    def __init__(self, image_0, image_1, params):
        assert image_0.shape == image_1.shape == params['shape'], "image shape must match"

        self.img_0 = image_0
        self.img_1 = image_1
        self.shape = params['shape']
        self.indexes = params['indexes']
        self.length = self.shape[0] * self.shape[1]
        self.coo_hess = CoordinateHessian(self.shape)
        self.linear_term = np.zeros(2 * self.length)
        self.__build_optical_flow_mat()

    def __build_optical_flow_mat(self):

        equation_params_01 = get_image_diff_equation(self.img_0, self.img_1)
        self.__add_to_hessian(equation_params_01)
        self.__add_to_linear_term(equation_params_01['linear'])

        equation_params_10 = get_image_diff_equation(self.img_1, self.img_0)
        self.__add_to_hessian(equation_params_10)
        self.__add_to_linear_term(-equation_params_10['linear'])

    def __add_to_linear_term(self, lin_t):

        self.linear_term[:self.length] += lin_t[0].ravel()
        self.linear_term[self.length:] += lin_t[1].ravel()

    def __add_to_hessian(self, equation_params):

        self.coo_hess.add_hess(self.indexes[:self.length], self.indexes[:self.length], equation_params['diagonal'][0].ravel().tolist())
        self.coo_hess.add_hess(self.indexes[self.length:], self.indexes[self.length:], equation_params['diagonal'][1].ravel().tolist())
        self.coo_hess.add_hess(self.indexes[:self.length], self.indexes[self.length:], equation_params['off_diagonal'].ravel().tolist())
        self.coo_hess.add_hess(self.indexes[self.length:], self.indexes[:self.length], equation_params['off_diagonal'].ravel().tolist())

class SmoothingBuilder:

    def __init__(self, params):
        self.params = params
        self.shape = params['shape']
        self.coo_hess = CoordinateHessian(self.shape)
        self.__build_derivatives()
        self.__build_HornSchnck_smoothness_term()
        self.__build_divergence_smoothness_term()
        self.__build_curl_smoothness_term()

    def __build_HornSchnck_smoothness_term(self):
        smth = self.params['smoothing']['HornSchunck']
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ii, self.d_ii, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ji, self.d_ji, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ij, self.d_ij, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_jj, self.d_jj, smth, self.shape))

    def __build_divergence_smoothness_term(self):
        smth = self.params['smoothing']['divergence']
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ii, self.d_ii, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ii, self.d_jj, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_jj, self.d_ii, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_jj, self.d_jj, smth, self.shape))

    def __build_curl_smoothness_term(self):
        smth = self.params['smoothing']['curl']
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ij, self.d_ij, smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ij, self.d_ji, -smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ji, self.d_ij, -smth, self.shape))
        self.coo_hess.add(get_coo_hess_of_two_operators(self.d_ji, self.d_ji, smth, self.shape))

    def __build_derivatives(self):
        index_mat_i = np.array(self.params['indexes'][:self.params['length']]).reshape(self.params['shape'])
        self.d_ii = get_square_operator(index_mat_i, coef = (-1, -1, 1, 1))
        self.d_ji = get_square_operator(index_mat_i, coef = (-1, 1, -1, 1))

        index_mat_j = np.array(self.params['indexes'][self.params['length']:]).reshape(self.params['shape'])
        self.d_ij = get_square_operator(index_mat_j, coef=(-1, -1, 1, 1))
        self.d_jj = get_square_operator(index_mat_j, coef=(-1, 1, -1, 1))

class CoordinateHessian:

    def __init__(self, shape):

        length = 2*shape[0]*shape[1]
        self.shape = (length, length)
        self.coo_mat = self.__get_coo_mat([], [], [])

    def __call__(self, *args, **kwargs):
        return self.coo_mat

    def add(self, other):
        self.coo_mat += other.coo_mat

    def add_hess(self, i, j, vals, **kargs):
        assert len(i) == len(j) == len(vals), 'all input lengths must be equal'
        self.coo_mat += self.__get_coo_mat(i, j, vals)

    def __get_coo_mat(self, i, j, vals):
        return coo_matrix((vals, (i, j)), shape=self.shape)

def get_image_diff_equation(image_0, image_1):

    diff = image_0 - image_1
    grad_0, grad_1 = np.gradient(image_0)
    lin_0 = diff*grad_0
    lin_1 = diff*grad_1
    diag_0 = grad_0*grad_0
    diag_1 = grad_1*grad_1
    off_diag = grad_0*grad_1
    return {'linear': np.array([lin_0, lin_1]), 'diagonal': [diag_0, diag_1], 'off_diagonal': off_diag}

def get_square_operator(index_mat, coef = (1,1,1,1)):

    ans = dict()
    length = index_mat[1:, 1:].size
    ans['00'] = {'index': index_mat[:-1, :-1].ravel().tolist(), 'value': coef[0] * np.ones(length)}
    ans['01'] = {'index': index_mat[:-1, 1: ].ravel().tolist(), 'value': coef[1] * np.ones(length)}
    ans['10'] = {'index': index_mat[1: , :-1].ravel().tolist(), 'value': coef[2] * np.ones(length)}
    ans['11'] = {'index': index_mat[1: , 1: ].ravel().tolist(), 'value': coef[3] * np.ones(length)}
    return ans

def get_coo_hess_of_two_operators(op0, op1, coef, shape):
    coo_hess = CoordinateHessian(shape=shape)
    for val0 in op0.values():
        for val1 in op1.values():
            coo_hess.add_hess(val0['index'], val1['index'], coef*val0['value']*val1['value'])

    return coo_hess