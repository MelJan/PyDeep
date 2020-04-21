""" This module provides functions for measuring like time measuring for executed code.

    :Version:
        1.1.0

    :Date:
        19.03.2017

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

"""
import datetime
import time
import numpy as numx


def print_progress(step,
                   num_steps,
                   gauge=False,
                   length=50,
                   decimal_place=1):
    """ Prints the progress of a system at state 'step'.

    :param step: Current step between 0 and num_steps-1.
    :type step: int

    :param num_steps: Total number of steps.
    :type num_steps: int

    :param gauge: If true prints a gauge
    :type gauge: bool

    :param length: Length of the gauge (in number of chars)
    :type length: int

    :param decimal_place: Number of decimal places to display.
    :type decimal_place: int
    """
    # jump to line beginning
    print('\r')
    if gauge:
        # print gauge
        print('=' * int(step * length / num_steps) + '>' + '.' * int(length - step * length / num_steps))
    # Define where to start printing the difits
    percent_format = '%'+str(3+decimal_place+numx.sign(decimal_place))+'.'+str(decimal_place)+'f%%'
    # Print formated percentage
    percent = (step * 100.0 / num_steps)
    print(percent_format % percent)
    if step == num_steps:
        print("")


class Stopwatch(object):
    """ This class provides a stop watch for measuring the execution time of code.

    """

    def __init__(self):
        """ Constructor sets the starting time to the current time.

            :Info: Will be overwritten by calling start()!

        """
        self.__start_time = datetime.datetime.now()
        self.__end_time = None
        self.__interval = 0.0
        self.__t_start = time.time()
        self.__t_last = time.time()

    def start(self):
        """ Sets the starting time to the current time.

        """
        self.__start_time = datetime.datetime.now()
        self.__end_time = None
        self.__interval = 0.0
        self.__t_start = time.time()
        self.__t_last = time.time()

    def pause(self):
        """ Pauses the time measuring.

        """
        t_temp = time.time()
        self.__interval += t_temp - self.__t_last

    def resume(self):
        """ Resumes the time measuring.

        """
        self.__t_last = time.time()

    def update(self, factor=1.0):
        """ | Updates the internal variables.
            | Factor can be used to sum up not regular events in a loop:
            | Lets assume you have a loop over 100 sets and only every 10th
            | step you execute a function, then use update(factor=0.1) to
            | measure it.

        :param factor: Sums up factor*current interval
        :type factor: float
        """
        t_temp = time.time()
        self.__interval += factor*(t_temp - self.__t_last)
        self.__t_last = t_temp

    def end(self):
        """ Stops/ends the time measuring.

        """
        self.update()
        self.__end_time = datetime.datetime.now()

    def get_start_time(self):
        """ Returns the starting time.

        :return: Starting time:
        :rtype: datetime
        """
        return self.__start_time

    def get_end_time(self):
        """ Returns the end time.

        :return: End time:
        :rtype: datetime
        """
        return self.__end_time

    def get_interval(self):
        """ Returns the current interval.

        :return: Current interval:
        :rtype: timedelta
        """
        self.update()
        return datetime.timedelta(0, self.__interval)

    def get_expected_end_time(self,
                              iteration,
                              num_iterations):
        """ Returns the expected end time.

        :param iteration: Current iteration
        :type iteration: int

        :param num_iterations: Total number of iterations.
        :type num_iterations: int

        :return: Expected end time.
        :rtype: datetime
        """
        return self.__start_time + self.get_expected_interval(iteration, num_iterations)

    def get_expected_interval(self,
                              iteration,
                              num_iterations):
        """ Returns the expected interval/Time needed till ending.

        :param iteration: Current iteration
        :type iteration: int

        :param num_iterations: Total number of iterations.
        :type num_iterations: int

        :return: Expected interval.
        :rtype: timedelta
        """
        self.update()
        expected_time = self.__interval+(num_iterations-iteration)*(self.__interval / iteration)
        return datetime.timedelta(0, expected_time)
