''' Test module for RBM trainer.
        
    :Version:
        1.1.0

    :Date:
        29.03.2017

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
import numpy as numx
import unittest
import sys

from pydeep.misc.toyproblems import generate_bars_and_stripes_complete
import pydeep.rbm.model as Model
import pydeep.rbm.trainer as Trainer
import pydeep.base.numpyextension as npExt
import pydeep.rbm.estimator as Estimator

print("\n... pydeep.rbm.trainer.py")


class TestTrainer(unittest.TestCase):
    epsilon = 0.00001

    def test__calculate_centered_gradient(self):
        sys.stdout.write('RBM Trainer -> Performing calculate_centered_gradient test ... ')
        sys.stdout.flush()
        numx.random.seed(42)
        gradsOrg = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, -1.0]])]
        visible_offsets = numx.array([[0.0, 0.0]])
        hidden_offsets = numx.array([[0.0, 0.0]])
        gradsCen = Trainer.CD._calculate_centered_gradient(gradsOrg, visible_offsets, hidden_offsets)
        assert numx.all(numx.abs(gradsOrg[0] - gradsCen[0]) < self.epsilon)
        assert numx.all(numx.abs(gradsOrg[1] - gradsCen[1]) < self.epsilon)
        assert numx.all(numx.abs(gradsOrg[2] - gradsCen[2]) < self.epsilon)
        visible_offsets = numx.array([[0.5, 0.5]])
        hidden_offsets = numx.array([[0.1, 0.2]])
        target = [numx.array([[1.1, 0.6], [0.2, 0.5]]), numx.array([[1.77, 3.88]]), numx.array([[-2.65, -1.55]])]
        gradsCen = Trainer.CD._calculate_centered_gradient(gradsOrg, visible_offsets, hidden_offsets)
        assert numx.all(numx.abs(target[0] - gradsCen[0]) < self.epsilon)
        assert numx.all(numx.abs(target[1] - gradsCen[1]) < self.epsilon)
        assert numx.all(numx.abs(target[2] - gradsCen[2]) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test___init__(self):
        sys.stdout.write('RBM Trainer -> Performing init test ... ')
        sys.stdout.flush()
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 2)
        trainer = Trainer.CD(model, data)
        assert numx.all(numx.abs(trainer.visible_offsets - data.mean(axis=0).reshape(1, data.shape[1])) < self.epsilon)
        trainer = Trainer.CD(model, None)
        assert numx.all(numx.abs(trainer.visible_offsets - 0.5) < self.epsilon)
        assert numx.all(numx.abs(trainer.hidden_offsets - 0.5) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_adapt_gradient(self):
        sys.stdout.write('RBM Trainer -> Performing adapt_gradient test ... ')
        sys.stdout.flush()
        # Test identity
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model, data[:, 0:2])
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [((pos_gradients[0] - neg_gradients[0]) / batch_size),
                  ((pos_gradients[1] - neg_gradients[1]) / batch_size),
                  ((pos_gradients[2] - neg_gradients[2]) / batch_size)]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # Test momentum , when prev grad was zero
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model, data[:, 0:2])
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.5, 0.5, 0.5],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [((pos_gradients[0] - neg_gradients[0]) / batch_size),
                  ((pos_gradients[1] - neg_gradients[1]) / batch_size),
                  ((pos_gradients[2] - neg_gradients[2]) / batch_size)]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # Test L1 Norm
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.5,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [numx.array([[-0.1, -0.5], [1., 0.1]]), numx.array([[1., 5.]]), numx.array([[-3., 3.]])]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # test L2 Norm
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.5,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [numx.array([[-1.22862968, 1.4092448], [2.05873296, 2.15099481]]), numx.array([[1., 5.]]),
                  numx.array([[-3., 3.]])]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # test Sparseness
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=1.0,
                                desired_sparseness=0.01,
                                mean_hidden_activity=numx.array([0.6, 0.7]),
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [numx.array([[0.4, 0.], [0.5, 0.6]]), numx.array([[1., 5.]]), numx.array([[-3.59, 2.31]])]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # test centered gradient
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=numx.array([[0.8, 0.5]]),
                                hidden_offsets=numx.array([[0.3, 0.2]]),
                                use_centered_gradient=True,
                                restrict_gradient=False,
                                restriction_norm=None)
        target = [numx.array([[2.5, -2.6], [0.5, -1.9]]), numx.array([[0.77, 5.23]]), numx.array([[-5.25, 6.03]])]
        assert numx.all(numx.abs(trainer.parameter_updates[0] - target[0]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[1] - target[1]) < self.epsilon)
        assert numx.all(numx.abs(trainer.parameter_updates[2] - target[2]) < self.epsilon)
        # test restriction
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=0.01,
                                restriction_norm='Cols')
        norm = npExt.get_norms(trainer.parameter_updates[0], 0)
        assert numx.all(numx.abs(norm - 0.01) < self.epsilon)
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=0.01,
                                restriction_norm='Rows')
        norm = npExt.get_norms(trainer.parameter_updates[0], 1)
        assert numx.all(numx.abs(norm - 0.01) < self.epsilon)
        pos_gradients = [numx.array([[0.3, 0.5], [-0.4, 0.8]]), numx.array([[2.0, 4.0]]), numx.array([[-2.0, 2.0]])]
        neg_gradients = [numx.array([[-0.1, 0.5], [-.9, 0.2]]), numx.array([[1.0, -1.0]]), numx.array([[1.0, -1.0]])]
        batch_size = 1
        model = Model.BinaryBinaryRBM(2, 2)
        trainer = Trainer.CD(model)
        trainer._adapt_gradient(pos_gradients=pos_gradients,
                                neg_gradients=neg_gradients,
                                batch_size=batch_size,
                                epsilon=[1, 1, 1],
                                momentum=[0.0, 0.0, 0.0],
                                reg_l1norm=0.0,
                                reg_l2norm=0.0,
                                reg_sparseness=None,
                                desired_sparseness=None,
                                mean_hidden_activity=None,
                                visible_offsets=0.0,
                                hidden_offsets=0.0,
                                use_centered_gradient=False,
                                restrict_gradient=0.01,
                                restriction_norm='Mat')
        norm = npExt.get_norms(trainer.parameter_updates[0], None)
        assert numx.all(numx.abs(norm - 0.01) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test__train(self):
        sys.stdout.write('RBM Trainer -> Performing train test ... ')
        sys.stdout.flush()
        # Zero offsets
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='00')
        assert numx.all(numx.abs(trainer.visible_offsets) < self.epsilon)
        assert numx.all(numx.abs(trainer.hidden_offsets) < self.epsilon)
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        assert numx.all(numx.abs(model.ov) < self.epsilon)
        assert numx.all(numx.abs(model.oh) < self.epsilon)

        # Data offsets
        target_h = numx.array([[0.23645209, 0.77365328]])
        target_v = numx.mean(data, 0)
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='DD')
        assert numx.all(numx.abs(trainer.visible_offsets - target_v) < self.epsilon)
        assert numx.all(numx.abs(trainer.hidden_offsets - target_h) < self.epsilon)
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='DD')
        assert numx.all(numx.abs(model.ov - target_v) < self.epsilon)
        assert numx.all(numx.abs(model.oh - target_h) < self.epsilon)
        # Model offsets
        target_v = numx.array([[0.875, 0.625, 0.125, 0.625]])
        target_h = numx.array([[0.22522563, 0.92149333]])
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='MM')
        assert numx.all(numx.abs(trainer.visible_offsets - target_v) < self.epsilon)
        assert numx.all(numx.abs(trainer.hidden_offsets - target_h) < self.epsilon)
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='MM')
        assert numx.all(numx.abs(model.ov - target_v) < self.epsilon)
        assert numx.all(numx.abs(model.oh - target_h) < self.epsilon)
        # Average offsets
        target_v = numx.array([[0.6875, 0.5625, 0.3125, 0.5625]])
        target_h = numx.array([[[[0.23083886, 0.8475733]]]])
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='AA')
        assert numx.all(numx.abs(trainer.visible_offsets - target_v) < self.epsilon)
        assert numx.all(numx.abs(trainer.hidden_offsets - target_h) < self.epsilon)
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 2, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='AA')
        assert numx.all(numx.abs(model.ov - target_v) < self.epsilon)
        assert numx.all(numx.abs(model.oh - target_h) < self.epsilon)
        print('successfully passed!')
        sys.stdout.flush()

    def test_GD(self):
        sys.stdout.write('RBM Trainer -> Performing GD test ... ')
        sys.stdout.flush()
        bbrbmBestLLPossible = -1.732867951
        # Test normal RBM
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=5000,
                      epsilon=0.9,
                      update_visible_offsets=0.0,
                      update_hidden_offsets=0.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.3)
        # Test True gradient centering
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.GD(model, data)
        trainer.train(data=data,
                      num_epochs=1000,
                      epsilon=0.9,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 0.1)
        print('successfully passed!')
        sys.stdout.flush()

    def test_CD(self):
        sys.stdout.write('RBM Trainer -> Performing CD test ... ')
        sys.stdout.flush()
        bbrbmBestLLPossible = -1.732867951
        thres = 0.2
        epochs = 10000
        # Test normal RBM
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.0,
                      update_hidden_offsets=0.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 2 * thres)
        # Test True gradient centering
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.CD(model, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        print('successfully passed!')
        sys.stdout.flush()

    def test_PCD(self):
        sys.stdout.write('RBM Trainer -> Performing PCD test ... ')
        sys.stdout.flush()
        bbrbmBestLLPossible = -1.732867951
        thres = 0.3
        epochs = 20000
        # Test normal RBM
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.0,
                      update_hidden_offsets=0.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 3 * thres)
        # Test True gradient centering
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PCD(model, 8, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.01,
                      k=2,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        print('successfully passed!')
        sys.stdout.flush()

    def test_PT(self):
        sys.stdout.write('RBM Trainer -> Performing PT test ... ')
        sys.stdout.flush()
        bbrbmBestLLPossible = -1.732867951
        thres = 0.2
        epochs = 5000
        # Test normal RBM
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.0,
                      update_hidden_offsets=0.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 2 * thres)
        # Test True gradient centering
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.PT(model, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        print('successfully passed!')
        sys.stdout.flush()

    def test_IPT(self):
        sys.stdout.write('RBM Trainer -> Performing IPT test ... ')
        sys.stdout.flush()
        bbrbmBestLLPossible = -1.732867951
        thres = 0.2
        epochs = 5000
        # Test normal RBM
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.0,
                      update_hidden_offsets=0.0,
                      use_centered_gradient=False,
                      offset_typ='00')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < 2 * thres)
        # Test True gradient centering
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=False,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=1.0,
                      update_hidden_offsets=1.0,
                      use_centered_gradient=True,
                      offset_typ='DD')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='MM')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        numx.random.seed(42)
        data = generate_bars_and_stripes_complete(2)
        data = numx.vstack((data[0], data, data[5]))
        model = Model.BinaryBinaryRBM(4, 4, data)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=False,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        # Test True gradient centered gradient
        numx.random.seed(42)
        model = Model.BinaryBinaryRBM(4, 4, data, initial_visible_offsets=0.0, initial_hidden_offsets=0.0)
        trainer = Trainer.IPT(model, 8, 10, data)
        trainer.train(data=data,
                      num_epochs=epochs,
                      epsilon=0.1,
                      update_visible_offsets=0.1,
                      update_hidden_offsets=0.1,
                      use_centered_gradient=True,
                      offset_typ='AA')
        LL = numx.mean(
            Estimator.log_likelihood_v(model, Estimator.partition_function_factorize_h(model, batchsize_exponent=3),
                                       data))
        assert numx.all(numx.abs(LL - bbrbmBestLLPossible) < thres)
        print('successfully passed!')
        sys.stdout.flush()
