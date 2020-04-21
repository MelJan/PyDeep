""" Provides a thread/script pooling mechanism based on ssh + screen.

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
import paramiko
from encryptedpickle import encryptedpickle
import cPickle
import datetime


class SSHConnection(object):
    """ Handles a SSH connection.
    """

    def __init__(self, hostname, username, password, max_cpus_usage=2):
        """ Constructor takes hostname, username, password.

        :param hostname: Hostname or address of host.
        :type hostname: string

        :param username: SSH username.
        :type username: string

        :param password: SSH password.
        :type password: string

        :param max_cpus_usage: Maximal number of cores to be used
        :type max_cpus_usage: int
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.username = username
        self.password = password
        self.hostname = hostname
        self.architecture = "Unknown"
        self.cpu_count = 0
        self.cpu_speed = 0
        self.max_cpus_usage = max_cpus_usage
        self._free_cpus_last_request = 0.0
        self.memory_size = 0
        self.raw_cpu_info = {}
        self.raw_memory_info = {}
        self.is_connected = False

    def encrypt(self, password):
        """ Encrypts the connection object.

        :param password: Encryption password
        :type password: string

        :return: Encrypted object
        :rtype: object
        """
        passphrases = {0: password}
        encoder = encryptedpickle.EncryptedPickle(signature_passphrases=passphrases, encryption_passphrases=passphrases)
        return encoder.seal(self)

    @classmethod
    def decrypt(cls, connection, password):
        """ Decrypts a connection object and returns it

        :param connection: SSHConnection to be decrypted
        :type connection: string

        :param password: Encryption password
        :type password: string

        :return: Decrypted object
        :rtype: SSHConnection
        """
        passphrases = {0: password}
        encoder = encryptedpickle.EncryptedPickle(signature_passphrases=passphrases, encryption_passphrases=passphrases)
        return encoder.unseal(connection)

    def connect(self):
        """ Connects to the server.

        :return: turns True is the connection was sucessful
        :rtype: bool

        """
        self.disconnect()
        try:
            self.client.connect(hostname=self.hostname, username=self.username, password=self.password)
            self.is_connected = True
        except:
            self.is_connected = False
        return self.is_connected

    def disconnect(self):
        """ Disconnects from the server.

        """
        self.client.close()
        self.is_connected = False

    def execute_command(self, command):
        """ Executes a command on the server and returns stdin, stdout, and stderr

        :param command: Command to be executed.
        :type command: string

        :return: stdin, stdout, and stderr
        :rtype: list
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            return self.client.exec_command(command)
        else:
            return None, None, None

    def execute_command_in_screen(self, command):
        """ Executes a command in a screen on the server which is automatically detached and returns stdin, stdout, \
            and stderr. Screen closes automatically when the job is
            done.

        :param command: Command to be executed.
        :type command: string

        :return: stdin, stdout, and stderr
        :rtype: list
        """
        return self.execute_command(command='screen -d -m ' + command)

    def renice_processes(self, value):
        """ Renices all processes.

        :param value: The New nice value -40 ... 20
        :type value: int or string

        :return: stdin, stdout, and stderr
        :rtype: list
        """
        return self.execute_command('renice ' + str(value) + ' -u ' + self.username)

    def kill_all_processes(self):
        """ Kills all processes.

        :return: stdin, stdout, and stderr
        :rtype: list
        """
        return self.execute_command('killall -u ' + self.username)

    def kill_all_screen_processes(self):
        """ Kills all acreen processes.

        :return: stdin, stdout, and stderr
        :rtype: list
        """
        return self.execute_command('killall -15 screen')

    def get_server_info(self):
        """ Get the server info like number of cpus, meomory size and stores it in the corresponding variables.

        :return: online or offline FLAG
        :rtype: string
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            # Get CPU info
            _, stdout, _ = self.execute_command('lscpu')
            stdout = stdout.readlines()
            for item in stdout:
                kvp = item.split(':')
                self.raw_cpu_info[kvp[0]] = kvp[1].replace(' ', '')

            self.architecture = self.raw_cpu_info['CPU op-mode(s)']
            self.cpu_count = int(self.raw_cpu_info['CPU(s)'])
            self.cpu_speed = int(self.raw_cpu_info['Thread(s) per core'])

            # Get memory info
            _, stdout, _ = self.execute_command('free -m')
            stdout = stdout.readlines()
            keys = stdout[0].split()
            values = stdout[1].split()
            for i in range(len(keys)):
                self.raw_memory_info[keys[i]] = values[i + 1]
            self.memory_size = int(self.raw_memory_info['total'])
            return 'online '
        return 'offline'

    def get_server_load(self):
        """ Get the current cpu and memory of the server.

        :return: | Average CPU(s) usage last  1 min,
                 | Average CPU(s) usage last  5 min,
                 | Average CPU(s) usage last 15 min,
                 | Average memory usage,
        :rtype: list
        """
        if not self.is_connected:
            self.connect()
        if self.is_connected:
            # Get CPU info
            if self.cpu_count == 0:
                self.get_server_info()
            _, stdout, _ = self.execute_command('cat /proc/loadavg')
            cpu_load = str.split(str(stdout.readlines()[0]))[0:3]
            # Get memory info
            _, stdout, _ = self.execute_command('free -m')
            mem_load = int(stdout.readlines()[1].split()[2])
            self._free_cpus_last_request = self.cpu_count - float(cpu_load[0])
            return float(cpu_load[0]), float(cpu_load[1]), float(cpu_load[2]), mem_load
        else:
            return None, None, None, None

    def get_number_users_processes(self):
        """ Gets number of processes of the user on the server.

        :return: number of processes
        :rtype: int or None
        """
        res = self.execute_command('ps aux | grep -c ' + self.username)[1]
        if res is None:
            return None
        else:
            return int(res.readlines()[0])

    def get_number_users_screens(self):
        """ Gets number of users screens on the server.

        :return: number of users screens on the server.
        :rtype: int or None
        """
        res1 = self.execute_command('screen -ls | grep -c Attached')[1]
        if res1 is None:
            return None
        else:
            res2 = self.execute_command('screen -ls | grep -c Detached')[1]
            return int(res1.readlines()[0]) + int(res2.readlines()[0])


class SSHJob(object):
    """ Handles a SSH JOB.
    """

    def __init__(self, command, num_threads=1, nice=19):
        """ Saves the encrypted serverlist to path.

        :param command: Command to be extecuted.
        :type command: string

        :param num_threads: Number of threads the job needs.
        :type num_threads: int

        :param nice: Nice value for this job.
        :type nice: int
        """
        self.command = command
        self.num_threads = num_threads
        self.nice = nice


class SSHPool(object):
    """ Handles a pool of servers and allows to distribute jobs over the pool.
    """

    def __init__(self, servers):
        """ Constructor takes a list of SSHConnections.

        :param servers: List of SSHConnections.
        :type servers: list
        """
        self.servers = servers
        self.log = []

    def save_server(self, path, password):
        """ Saves the encrypted serverlist to path.

        :param path: Path and filename
        :type path: string

        :param password: Encrption password
        :type password: string
        """
        encrypted_server_list = []
        for s in self.servers:
            encrypted_server_list.append(s.encrypt(password))
        try:
            cPickle.dump(open(path, 'w'))
            self.log.append(str(datetime.datetime.now()) + ' Server save to ' + path)
        except:
            raise Exception("-> File writing Error: ")

    def load_server(self, path, password, append=True):
        """

        :param path: Path and filename.
        :type path: string

        :param password: Encrption password.
        :type password: string

        :param append: If true, servers get append to list, if false server list gets replaced.
        :type append: bool
        """
        try:
            encrypted_server_list = cPickle.load(open(path, 'r'))
        except:
            raise Exception("-> File reading Error!")

        if append is False:
            self.servers.clear()
        try:
            for s in encrypted_server_list:
                self.servers.append(s.decrypt(password))
            self.log.append(str(datetime.datetime.now()) + ' Server loaded from ' + path)
        except:
            raise Exception("Wrong password!")

    def execute_command(self, host, command):
        """ Executes a command on a given server servers.

        :param host: Hostname or connection object
        :type host: string or SSHConnection

        :param command: Command to be executed
        :type command: string

        :return:
        :rtype:
        """
        if isinstance(host, SSHConnection):
            s = host
        else:
            s = self.servers[host]
        output = 'offline'
        if s.connect():
            output = s.execute_command(command)
            self.log.append(str(datetime.datetime.now()) + ' Command ' + command + ' executed on ' + host.hostname)
        s.disconnect()
        return output

    def execute_command_in_screen(self, host, command):
        """ Executes a command in a screen on a given server servers.

        :param host: Hostname or connection object
        :type host: string or SSHConnection

        :param command: Command to be executed
        :type command: string

        :return: list of all stdin, stdout, and stderr
        :rtype: list
        """
        if isinstance(host, SSHConnection):
            s = host
        else:
            self.servers[host]
        output = 'offline'
        if s.connect():
            output = s.execute_command_in_screen(command)
            self.log.append(
                str(datetime.datetime.now()) + ' Command in screen ' + command + ' executed on ' + host.hostname)
        s.disconnect()
        return output

    def broadcast_command(self, command):
        """ Executes a command an all servers.

        :param command: Command to be executed
        :type command: string

        :return: list of all stdin, stdout, and stderr
        :rtype: list
        """
        output = {}
        for s in self.servers:
            if s.connect():
                output[s.hostname] = s.execute_command(command)
            else:
                output[s.hostname] = 'offline'
            s.disconnect()
        self.log.append(str(datetime.datetime.now()) + ' Broadcast ' + command + ' send to all servers')
        return output

    def broadcast_kill_all(self):
        """ Kills all processes on the server of the corresponding user.

        :return: list of all stdin, stdout, and stderr
        :rtype: list
        """
        output = {}
        for s in self.servers:
            if s.connect():
                output[s.hostname] = s.kill_all_processes()
            else:
                output[s.hostname] = 'offline'
            s.disconnect()
        self.log.append(str(datetime.datetime.now()) + ' Kill all broadcast send to all servers')
        return output

    def broadcast_kill_all_screens(self):
        """ Kills all screens on the server of the corresponding user.

        :return: list of all stdin, stdout, and stderr
        :rtype: list
        """
        self.broadcast_command('killall -15 screen')

    def distribute_jobs(self, jobs, status=False, ignore_load=False, sort_server=True):
        """ Distributes the jobs over the servers.

        :param jobs: List of SSHJobs to be executeed on the servers.
        :type jobs: string or SSHConnection

        :param status: If true prints info about which job was started on which server.
        :type status: bool

        :param ignore_load: If true starts the job without caring about the current load.
        :type ignore_load: bool

        :param sort_server: If True Servers will be sorted by load.
        :type sort_server: bool

        :return: List of all started jobs and list of all remaining jobs
        :rtype: list, list
        """
        self.get_servers_status()
        # Sort Server by free capacity
        if sort_server is True:
            self.servers.sort(key=lambda x: x._free_cpus_last_request, reverse=True)
        # Sort Server by num_threads
        jobs.sort(key=lambda x: x.num_threads, reverse=True)
        # List of started jobs
        started_job = []
        # Loop over Server
        for server in self.servers:
            if status:
                print("Server: " + server.hostname)
            server.connect()
            server.get_server_info()

            if ignore_load is True:
                num_free_cores = server.max_cpus_usage
            else:
                num_free_cores = server.cpu_count - server.get_server_load()[0]
                if num_free_cores > server.max_cpus_usage:
                    num_free_cores = server.max_cpus_usage
            if status:
                print("\tFree cores: " + str(num_free_cores))
            if status:
                print("\tJobs started:")
            started_job_index = []
            for j in range(len(jobs)):
                if num_free_cores < 1:
                    break
                threads_to_use = jobs[j].num_threads
                if threads_to_use <= num_free_cores:
                    _, _, _ = server.execute_command(jobs[j].command)
                    self.log.append(
                        str(datetime.datetime.now()) + ' Job ' + jobs[j].command + ' started on ' + server.hostname)
                    if status:
                        print("\t\t " + jobs[j].command)
                    started_job_index.append(jobs[j])
                    started_job.append(jobs[j])
                    num_free_cores -= threads_to_use

            # print started_job_index
            for j in range(len(started_job_index)):
                jobs.remove(started_job_index[j])
            if status:
                print("\tNow Free cores: " + str(num_free_cores))
            server.disconnect()
        return started_job, jobs

    def get_servers_status(self):
        """ Reads the status of all servers and returns it a list. Additionally print to the console if status == True.

        :return: list of header and list corresponding status information
        :rtype: list, list
        """
        results = []
        header = ['hostname       ',
                  'status         ',
                  'user processes ',
                  'user screens   ',
                  'sys load(%)1m  ',
                  'sys load(%)5m  ',
                  'sys load(%)15m ',
                  'used memory(%) ',
                  'free cpus 1min ',
                  'free cpus 5min ',
                  'free cpus 15min',
                  'free memory(MB)']
        for s in self.servers:
            if s.connect():
                load = s.get_server_load()
                processes = s.get_number_users_processes()
                screens = s.get_number_users_screens()
                results.append([s.hostname, 'online', processes, screens, 100.0 * load[0] / s.cpu_count,
                                100.0 * load[1] / s.cpu_count, 100.0 * load[2] / s.cpu_count,
                                100.0 * load[3] / float(s.memory_size), s.cpu_count - load[0], s.cpu_count - load[1],
                                s.cpu_count - load[2], s.memory_size - load[3]])
            else:
                results.append([s.hostname, 'offline', '-', '-', '-', '-', '-', '-', '-', '-', '-'])
            s.disconnect()
        results.sort(key=lambda x: x[9], reverse=True)
        for h in header:
            print str(h),
        print("")
        for r in results:
            for i in r:
                if isinstance(i, float):
                    print '%.2f\t\t' % i,
                else:
                    print str(i) + '\t\t',
            print("")
        return header, results

    def get_servers_info(self, status=True):
        """ Reads the status of all servers, the information is stored
            in the SSHConnection objects.
            Additionally print to the console if status == True.

        :param status: If true prints info.
        :type status: bool

        """
        # Add all zappas
        if status is True:
            print 'Hostname\tstatus\t\tCPU count\tMax CPU usage\tMemory size \tCPU speed \tCPU architecture'
        for s in self.servers:
            onoff = "offline"
            if s.connect():
                onoff = s.get_server_info()
                s.disconnect()
            if status is True:
                print s.hostname, '\t', onoff, '\t', s.cpu_count, '\t\t', s.max_cpus_usage, '\t\t', \
                    s.memory_size, '\t\t', s.cpu_speed, '\t\t', s.architecture,
