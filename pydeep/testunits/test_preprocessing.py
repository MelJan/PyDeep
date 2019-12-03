''' Test module for preprocessing methods.

    :Version:
        1.1.0

    :Date:
        04.04.2017

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2017 Jan Melchior

        This file is part of the Python library PyDeep.

        PyDeep is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
import sys
import unittest

import numpy as numx

from pydeep.base.numpyextension import get_norms, angle_between_vectors
from pydeep.preprocessing import binarize_data, rescale_data, remove_rows_means, remove_cols_means, STANDARIZER, PCA, \
    ZCA, ICA
from pydeep.misc.toyproblems import generate_2d_mixtures

print("\n... pydeep.preprocessing_PCA_ICA.py")


class Test_Preprocessing(unittest.TestCase):
    epsilon = 0.00001

    def test_binarize_data(self):
        print('Preprocessing -> Performing binarize_data test ...')
        sys.stdout.flush()
        data = binarize_data(numx.random.rand(100, 100))
        assert numx.sum(data != 0.0) + numx.sum(data != 1.0) == 10000
        data = binarize_data(numx.random.randn(100, 100))
        assert numx.sum(data != 0.0) + numx.sum(data != 1.0) == 10000
        print('successfully passed!')
        sys.stdout.flush()

    def test_rescale_data(self):
        print('Preprocessing -> Performing rescale_data test ...')
        sys.stdout.flush()
        data = rescale_data(numx.random.randn(100, 100), new_min=-2.0, new_max=4.0)
        assert numx.sum(data >= -2.0) + numx.sum(data <= 4.0) == 20000
        print('successfully passed!')
        sys.stdout.flush()

    def test_remove_rows_means(self):
        print('Preprocessing -> Performing remove_rows_means test ...')
        sys.stdout.flush()
        data = numx.random.randn(100, 1000)
        dataMean = numx.mean(data, axis=1).reshape(100, 1)
        MFdata, meanF = remove_rows_means(data, return_means=True)
        assert numx.all(numx.abs(dataMean - meanF) < self.epsilon)
        zeroMean = numx.mean(MFdata, axis=1)
        assert numx.all(numx.abs(zeroMean) < self.epsilon ** 2)
        print('successfully passed!')
        sys.stdout.flush()

    def test_remove_cols_means(self):
        print('Preprocessing -> Performing remove_cols_means test ...')
        sys.stdout.flush()
        data = numx.random.randn(100, 1000)
        dataMean = numx.mean(data, axis=0).reshape(1, 1000)
        MFdata, meanF = remove_cols_means(data, return_means=True)
        assert numx.all(numx.abs(dataMean - meanF) < self.epsilon)
        zeroMean = numx.mean(MFdata, axis=0)
        assert numx.all(numx.abs(zeroMean) < self.epsilon ** 2)
        print('successfully passed!')
        sys.stdout.flush()

    def test_STANDARIZER(self):
        print('Preprocessing -> Performing STANDARIZER test ...')
        sys.stdout.flush()
        data = numx.random.randn(100, 1000)
        dataMean = numx.mean(data, axis=0).reshape(1, 1000)
        dataStd = numx.std(data, axis=0).reshape(1, 1000)
        model = STANDARIZER(1000)
        model.train(data)
        assert numx.all(numx.abs(dataStd - model.standard_deviation) < self.epsilon)
        assert numx.all(numx.abs(dataMean - model.mean) < self.epsilon ** 2)
        projectedData = model.project(data)
        newMean = numx.mean(projectedData, axis=0).reshape(1, 1000)
        newStd = numx.std(projectedData, axis=0).reshape(1, 1000)
        assert numx.all(numx.abs(newStd - 1.0) < self.epsilon)
        assert numx.all(numx.abs(newMean < self.epsilon ** 2))
        unprojectedData = model.unproject(projectedData)
        newMean = numx.mean(unprojectedData, axis=0).reshape(1, 1000)
        newStd = numx.std(unprojectedData, axis=0).reshape(1, 1000)
        assert numx.all(numx.abs(newStd - dataStd) < self.epsilon)
        assert numx.all(numx.abs(newMean - dataMean) < self.epsilon ** 2)
        print('successfully passed!')
        sys.stdout.flush()

    def test_PCA(self):
        print('Preprocessing -> Performing PCA test ...')
        sys.stdout.flush()
        cov = numx.array([[1.0, 0.8], [0.8, 1.0]])
        mean = numx.array([1.0, -1.0])
        data = numx.random.multivariate_normal(mean, cov, 10000)
        model = PCA(data.shape[1], True)
        model.train(data)
        assert numx.all(numx.abs(data - model.unproject(model.project(data))) < self.epsilon)
        # Eigenvectors of the Gaussian point in 45/225 and 135/315
        # degree direction and the vectors only differ in sign
        target = 1.0 / numx.sqrt(2.0)
        target = numx.ones((2, 2)) * target
        model = PCA(data.shape[1], False)
        model.train(data)
        assert numx.all(numx.abs(numx.abs(model.projection_matrix) - target) < 0.01)
        assert numx.all(numx.abs(data - model.unproject(model.project(data))) < self.epsilon)
        assert model.project(data, 1).shape[1] == 1
        assert model.unproject(model.project(data, 1), 1).shape[1] == 1
        assert model.unproject(model.project(data), 1).shape[1] == 1
        assert model.unproject(model.project(data, 1)).shape[1] == 2
        print('successfully passed!')
        sys.stdout.flush()

    def test_ZCA(self):
        print('Preprocessing -> Performing ZCA test ...')
        sys.stdout.flush()
        cov = numx.array([[1.0, 0.8], [0.8, 1.0]])
        mean = numx.array([1.0, -1.0])
        data = numx.random.multivariate_normal(mean, cov, 10000)
        # Eigenvectors of the Gaussian point in 45/225 and 135/315
        # degree direction and the vectors only differ in sign
        target = 1.0 / numx.sqrt(2.0)
        target = numx.ones((2, 2)) * target
        model = ZCA(data.shape[1])
        model.train(data)
        # assert numx.all(numx.abs(numx.abs(model.projection_matrix)-target) < 0.01)
        assert numx.all(numx.abs(data - model.unproject(model.project(data))) < self.epsilon)
        assert model.project(data, 1).shape[1] == 1
        assert model.unproject(model.project(data, 1), 1).shape[1] == 1
        assert model.unproject(model.project(data), 1).shape[1] == 1
        assert model.unproject(model.project(data, 1)).shape[1] == 2
        print('successfully passed!')
        sys.stdout.flush()

    def test_ICA(self):
        print('Preprocessing -> Performing ICA test ...')
        sys.stdout.flush()
        fail = 0
        for _ in range(100):
            data, mixMat = generate_2d_mixtures(10000)
            zca = ZCA(data.shape[1])
            zca.train(data)
            data_zca = zca.project(data)
            mixMat_zca = zca.project(mixMat.T).T
            mixMat_zca /= get_norms(mixMat_zca, axis=None)
            model = ICA(2)
            model.train(data_zca, 10)
            assert numx.all(numx.abs(data_zca - model.unproject(model.project(data_zca))) < self.epsilon)

            # Check the angle between mixing matrix  vectors and resulting vectors of the projection matrix
            # Theorectially the amari distnace could be used but it seems only to be rotation and scal invariant
            # and not invariant through permutations of the colums.
            res = model.projection_matrix / get_norms(model.projection_matrix, axis=None)
            v1 = res
            v2 = res * numx.array([[-1, 1], [-1, 1]])
            v3 = res * numx.array([[1, -1], [1, -1]])
            v4 = -res
            res = numx.array([res[:, 1], res[:, 0]]).T
            v5 = res
            v6 = res * numx.array([[-1, 1], [-1, 1]])
            v7 = res * numx.array([[1, -1], [1, -1]])
            v8 = -res

            v1d = numx.max([angle_between_vectors(v1[0], mixMat_zca[0]), angle_between_vectors(v1[1], mixMat_zca[1])])
            v2d = numx.max([angle_between_vectors(v2[0], mixMat_zca[0]), angle_between_vectors(v2[1], mixMat_zca[1])])
            v3d = numx.max([angle_between_vectors(v3[0], mixMat_zca[0]), angle_between_vectors(v3[1], mixMat_zca[1])])
            v4d = numx.max([angle_between_vectors(v4[0], mixMat_zca[0]), angle_between_vectors(v4[1], mixMat_zca[1])])
            v5d = numx.max([angle_between_vectors(v5[0], mixMat_zca[0]), angle_between_vectors(v5[1], mixMat_zca[1])])
            v6d = numx.max([angle_between_vectors(v6[0], mixMat_zca[0]), angle_between_vectors(v6[1], mixMat_zca[1])])
            v7d = numx.max([angle_between_vectors(v7[0], mixMat_zca[0]), angle_between_vectors(v7[1], mixMat_zca[1])])
            v8d = numx.max([angle_between_vectors(v8[0], mixMat_zca[0]), angle_between_vectors(v8[1], mixMat_zca[1])])

            dist = numx.min([v1d, v2d, v3d, v4d, v5d, v6d, v7d, v8d])
            # biggest angle smaller than 5 degrees
            if dist > 5.0:
                fail += 1
        assert fail < 5
        print('successfully passed!')
        sys.stdout.flush()


if __name__ is "__main__":
    unittest.main()
