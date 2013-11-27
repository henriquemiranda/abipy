#!/usr/bin/env python
from __future__ import division, print_function

import abc
import os
import time
import subprocess
import collections
import readline
import pprint
import cStringIO as StringIO 
import json
import yaml
import socket
import paramiko

from termcolor import colored, cprint
from pymatgen.core.design_patterns import AttrDict
from pymatgen.util.string_utils import is_string


def straceback(color="red"):
    """
    Returns a string with the traceback.

    Use ANSII color formatting for output in terminal if color is not None.
    """
    import traceback
    s = traceback.format_exc()
    if color is not None:
        try:
            from termcolor import colored
            return colored(s, color)
        except ImportError:
            return s
    else:
        return s


def init_clusters(filepath="clusters.yml"):
    """Read the configuration paramenters from the YAML file clusters.yml."""
    with open(filepath, "r") as fh:
        conf = yaml.load(fh)

    # Global parameter (will be overwritten by cluster-specific params, if present).
    global_username = conf.get("username", None)
    global_workdir = conf.get("workdir", None)
    global_qtype = conf.get("qtype", None)

    d, clusters = conf["clusters"], {}

    for hostname, params in d.items():
        username = params.get("username", global_username)
        workdir = params.get("workdir", global_workdir)
        qtype = params.get("qtype", global_qtype)

        cls = Cluster.from_qtype(qtype)
        assert hostname not in clusters
        clusters[hostname] = cls(hostname, username, workdir)

    return clusters


class Cluster(object):
    """
    This object stores the basic parameters of the cluster that are needed to establish 
    SSH, SFTP connections. It also provides helper functions for monitoring the resource manager.
    This an abstract base class defining the interface that must be implemented 
    by the concrete subclasses. Every subclass must define the class attribute `qtype` so 
    that specified the type of resource manager installed on the cluster.

    A cluste has a working directory where we are going the generate and run Flows.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hostname, username, workdir):
        """
        Args:
            hostname:
                Name of the host.
            username:
                Username used to login on the cluster.
            workdir:
                Absolute path (on the remote host) where `AbinitFlows` will be produced.
        """
        self.hostname, self.username, = hostname, username
        self.workdir = workdir

        self.port = 22    # Port for SSH connection
        self.timeout = 30 # Timeout in seconds.

        # List of partitions available on the cluster.
        #self.partions = []

    @classmethod
    def from_qtype(cls, qtype):
        """Return the appropriate subclass from the queue type."""
        for subcls in cls.__subclasses__():
            if subcls.qtype == qtype:
                return subcls

        raise ValueError("Wrong value for qtype %s" % qtype)

    def __str__(self):
        """String representation."""
        return "%s@%s:%s [%s]" % (self.username, self.hostname, self.workdir, self.qtype)

    #def __del__(self):
    #    self.disconnect

    def path_inworkdir(self, basename):
        """Returns the absolute path in the working directory of the cluster."""
        return os.path.join(self.workdir, basename)

    def exists(self, path):
        """True if path exists on the cluster."""
        return self.ssh_exec("test -e %s" % path).return_code == 0

    def ping(self):
        """Ping the host."""
        cmd = ["ping", "-c1", "-W100", "-t1", self.hostname]
        ping_response = subprocess.Popen(cmd, stdout=subprocess.PIPE).stdout.read()
        return ping_response

    def prefix_str(self, s):
        """Add the name of the host to every line in s.splitlines()."""
        lines = ["[%s] %s" % (self.hostname, l) for l in s.splitlines()]

        if not lines:
            lines = ["[%s] (Empty)" % self.hostname]

        return "\n".join(lines)

    def disconnect(self):
        """Close the SSH and the SFPT client."""
        try:
            self.ssh.close()
        except:
            pass
        try:
            self.sftp.close()
        except:
            pass

    @property
    def has_slurm(self):
        """True if the host uses SLURM."""
        return self.qtype == "slurm"

    @property
    def has_sge(self):
        """True if the host uses SGE."""
        return self.qtype == "sge"

    def _ssh_connect(self):
        """Returns an instance of `MySSHClient`."""
        ssh_client = MySSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)

        ssh_client.connect(self.hostname, port=self.port, username=self.username)
        return ssh_client

    @property
    def ssh(self):
        """SSH client used to communicate with the remote host"""
        if hasattr(self, "_ssh") and self._ssh.is_connected:
            return self._ssh
        else:
            self._ssh = self._ssh_connect()
            return self._ssh

    def ssh_close(self):
        """Close the SSH connection."""
        if hasattr(self, "_ssh") and self._ssh.is_connected:
            self._ssh.close()

    def ssh_exec(self, command, timeout=None):
        """
        Execute command via ssh. Return `SSHResult` instance.
        """
        stdin, stdout, stderr = self.ssh.exec_command(command, timeout=timeout)
        return SSHResult(stdout, stderr)
         
    def _sftp_connect(self):
        """Returns an instance of `MySFTP Client`."""
        sftp = self.ssh.open_sftp()
        sftp.__class__ = MySFTPClient
        sftp._is_connected = True
        return sftp

    @property
    def sftp(self):
        """SFPT client used to communicate with the remote host"""
        if hasattr(self, "_sftp") and self._sftp.is_connected:
            return self._sftp
        else:
            self._sftp = self._sftp_connect()
            return self._sftp

    def sftp_close(self):
        """Close the SFPT connection."""
        if hasattr(self, "_sftp") and self._sftp.is_connected:
            self.sftp.close()

    #def sftp_put(self, localpath, remotepath, confirm, mode=None, close=True):
        #self.sftp.get(source, dest)
        #sftp = self.sftp
        #sftp.put(localpath=script, remotepath=remotepath, confirm=True)
        #sftp.chmod(remotepath, mode=0700)
        #if close: sftp.close()

    def exists(self, path):
        """True if path exists."""
        return self.ssh_exec("test -e %s" % path).return_code == 0

    def which(self, bin):
        """Execute `which bin` on the cluster."""
        return self.ssh_exec("which %s" % bin).out

    def read_file(self, path):
        return self.ssh_exec("cat %s" % path).out

    def make_workdir(self):
        """
        Create the working directory on the cluster if it does not exist.

        Returns:
            exit_status
        """
        self.make_dir(self.workdir)

    def make_dir(self, path):
        """
        Create a directory on the cluster if it does not exist.

        Returns:
            exit_status
        """
        if self.exists(path): return 0

        print("Will create remote dir %s" % path)
        return self.ssh_exec("mkdir %s" % path).return_code

    def rmdir(self, path):
        """Remove a directory on the cluster. Return exit status."""
        return self.ssh_exec("rm -rf %s" % path).return_code

    def invoke_shell(self, timeout=-1):
        """Returns an instance of `MyShell`."""
        shell = self.ssh.invoke_shell()
        shell.__class__ = MyShell
        shell.settimeout(timeout if timeout != -1 else self.timeout)
        return shell

    def invoke_abinit(self, *args):
        """Returns the output of `abinit args`."""
        shell = self.invoke_shell()

        result = shell.myexec("abinit %s" % " ".join(*args))
        if result.return_code != 0:
            # There are case in which abinit must be invoked via mpirun.
            result = shell.myexec("mpirun abinit %s" % " ".join(*args)).out

        return result

    def get_abinit_info(self, prefix=False):
        """
        Returns a dictionary with the Abinit version and the build info
        If prefix is True, hostname is prepended to each line
        """
        if not self.which("abinit"): 
            return {"version": "Cannot find abinit in $PATH"}

        return None
        # FIXME
        return dict(
            version=self.invoke_abinit("--version").out,
            build=self.invoke_abinit("--build").out
            )

    def abicheck(self, with_color=False):
        """
        Run abicheck.py on the cluster to make sure the environment is properly setup.

        args:
            with_color:
                True if ANSII colored string is wanted.

        Returns:
            human-readable string with the outcome of the test.
        """
        if not with_color:
            def colored(s, color): return s

        shell = self.invoke_shell()
        try:
            cmd = "abicheck.py"
            result = shell.myexec(cmd)

            if result.return_code:
                msg = colored("[FAILED]: ", "red") + "%s returned %d" % (cmd, result.return_code)
            else:
                msg = colored("[OK]", "blue")
                                                                                                  
        except socket.timeout:
            msg = colored("[FAILED]: ", "red") + "socket timeout"
                                                                                                  
        return msg

    def yaml_configurations(self):
        """List of dicts with the YAML configuration files used on the remote hosts."""

        conf_files = [
            "~/.abinit/abipy/taskmanager.yml",
            "~/.abinit/abipy/scheduler.yml",
        ]

        confs = []
        for f in conf_files:
            result = self.ssh_exec("cat %s" % f)
            confs.append(yaml.dump(yaml.load(result.out)))

        return confs

    @abc.abstractmethod
    def get_user_jobs(self, prefix=False):
        """
        Return a string with info on the jobs belonging to the user
        If prefix is True, hostname is prepended to each line
        """

    @abc.abstractmethod
    def get_all_jobs(self, prefix=False):
        """
        Return a string with info on all the jobs in the queue.
        If prefix is True, hostname is prepended to each line
        """

    @abc.abstractmethod
    def get_qinfo(self, prefix=False):
        """Return a string with info on the queue."""

    #@abc.abstractmethod
    #def get_qload(self, prefix=False):
    #    """Return a string with the load of the cluster"""


class SlurmCluster(Cluster):
    """A cluster with Slurm."""
    qtype = "slurm"

    def get_all_jobs(self, prefix=False):
        result = self.ssh_exec("squeue")
        return self.prefix_str(result.out) if prefix else result.out

    def get_user_jobs(self, prefix=False):
        result = self.ssh_exec("squeue -u %s" % self.username)
        return self.prefix_str(result.out) if prefix else result.out

    def get_qinfo(self, prefix=False):
        result = self.ssh_exec("sinfo")
        return self.prefix_str(result.out) if prefix else result.out

    #def get_qload(self, prefix=False):
    #    result = self.ssh_exec("sload")
    #    return self.prefix_str(result.out) if prefix else result.out


class SgeCluster(Cluster):
    """A cluster with SGE."""
    qtype = "sge"

    def get_all_jobs(self, prefix=False):
        result = self.ssh_exec('qstat -u "*"')
        return self.prefix_str(result.out) if prefix else result.out

    def get_user_jobs(self, prefix=False):
        result = self.ssh_exec("qstat -u %s" % self.username)
        return self.prefix_str(result.out) if prefix else result.out

    def get_qinfo(self, prefix=False):
        result = self.ssh_exec("qhost")
        return self.prefix_str(result.out) if prefix else result.out


class MySSHClient(paramiko.SSHClient):

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self, hostname, **kwargs):
        super(MySSHClient, self).connect(hostname, **kwargs)
        self._is_connected = False

    def close(self):
        if not self.is_connected: return
        super(MySSHClient, self).close()
        self._is_connected = False


class MySFTPClient(paramiko.SFTPClient):

    @property
    def is_connected(self):
        return self._is_connected

    def close(self):
        if not self.is_connected: return
        super(MySFTPClient, self).close()
        self._is_connected = False


class MyShell(paramiko.Channel):

    def myexec(self, s, with_exec=True):
        if with_exec and not s.startswith("exec"):
            s = "exec " + s

        if not s.endswith("\n"): s += "\n"

        # http://stackoverflow.com/questions/5342402/can-i-get-the-exit-code-of-a-command-executed-in-a-subshell-via-ssh
        self.sendall(s)

        #stdout = self.makefile()
        #stderr = self.makefile_stderr()
        # Wait for it.....
        #count = 0
        #while not self.recv_ready():
        #    time.sleep(1)
        #    count += 1
        #    if count >= 6:
        #        print('time out') #TODO: add exeption here 
        #time.sleep(10)

        #return_code = self.recv_exit_status()
        #print(return_code)
        #result = ""
        #while self.recv_ready():
        ##while self.recv_ready() and not self.exit_status_ready():
        #    o = self.recv(1024)
        #    #if not len(o): break
        #    #if o:
        #    print("o", o)
        #    result += o
        #self.settimeout(30)

        stdout, stderr = StringIO.StringIO(), StringIO.StringIO()

        #try: #TODO: add exeption here 
        # Need to read the entire buffer to caputure output
        while not self.exit_status_ready():

            if self.recv_ready():
                out_buff = self.recv(1024)
                while out_buff:
                    #print("Indside stdout", out_buff)
                    stdout.write(out_buff)
                    out_buff = self.recv(1024)


            if self.recv_stderr_ready():
                error_buff = self.recv_stderr(1024)
                while error_buff:
                    #print("Indside stderr", error_buff)
                    stderr.write(error_buff)
                    error_buff = self.recv_stderr(1024)

        #except socket.timeout:
        #  raise socket.timeout

        stdout.seek(0), stderr.seek(0)
        return SSHResult(stdout, stderr, return_code=self.recv_exit_status())

    def exit(self):
        self.sendall('exit\n')

        # Wait for it.....
        count = 0
        while not self.exit_status_ready():
            time.sleep(1)
            count += 1
            if count >= 6:
                print('time out')#TODO: add exeption here 

        return self.recv_exit_status()


class SSHResult(object):
    """
    This object stores the results of a command executed on the remote host.

    .. attributes:

        return_code:
            Return code of the command.
    """
    def __init__(self, stdout, stderr, return_code=None):
        self.stdout, self.stderr = stdout, stderr

        if return_code is None:
            self.return_code = stdout.channel.recv_exit_status()
        else:
            self.return_code = return_code

    def __str__(self):
        s  = "out: " + self.out 
        if self.err: s += colored("\nerr: " + self.err, "red")
        return s

    @property
    def succeeded(self):
        return self.return_code == 0

    @property
    def out(self):
        """Output of the SSH command."""
        try:
            return self._out
        except AttributeError:
            self._out = self.stdout.read()
            return self._out

    @property
    def err(self):
        """Stderr of the SSH command."""
        try:
            return self._err
        except AttributeError:
            self._err = self.stderr.read()
            return self._err

_ALL_CLUSTERS = init_clusters()

import cmd

class RunCommand(cmd.Cmd, object):
    """ Simple shell to run commands on the localhost """
    prompt='abife> '

    def __init__(self):
        cmd.Cmd.__init__(self)

        # Get a copy of the dict with the clusters.
        self.clusters = _ALL_CLUSTERS.copy()

        self.flows_db = FlowsDatabase()

    def emptyline(self):
        """Disable the repetition of the last command."""

    def onecmd(self, line):
        try:
            r = super(RunCommand, self).onecmd(line)
            if r and (self.can_exit() or raw_input('Exit anyway? [Y/n]:').lower().startswith("y")):
                 return True
            return False

        except:
            print(straceback())
            return False

    def can_exit(self):
        return False

    def do_exit(self, line):
        """Exit the interpreter. You can also use the Ctrl-D shortcut."""
        return True
                                                                                   
    do_EOF = do_exit

    def do_shell(self, line):
        """Execute shell command on the local host."""
        os.system(line)
                                        
    def complete_hostnames(self, text, line, begidx, endidx):
        """
        The complete_xxx method takes four arguments:

        Args:
            text:
                the string we are matching against, all returned matches must begin with it
                line is is the current input line
            begidx: 
                the beginning index in the line of the text being matched
            endidx: 
                the end index in the line of the text being matched

        It should return a list (possibly empty) of strings representing the possible completions. 
        The arguments begidx and endidx are useful when completion depends on the position of the argument.
        """
        return [i for i in self.clusters if i.startswith(text)]

    # Method called to complete an input line when no command-specific complete_*() method is available.
    completedefault = complete_hostnames

    def _select_clusters(self, line):
        """
        Returns the list of clusters corresponding to the input string
        If s is empy, the full list of clusters is returned.
        """
        if not line:
            return self.clusters.values()
        else:
            hosts = line.split()
            return filter(None, (self.clusters.get(h, None) for h in hosts))

    def do_disable_hosts(self, line):
        """
        Disable the specified list of hosts. 
        Syntax: disable_hosts host1 host2 ...
        """
        hosts = line.split()
        if not hosts:
            print(self.do_disable_hosts.__doc__)

        for h in hosts:
            self.clusters.pop(h, None)

    def do_reenable_hosts(self, line):
        """
        Renable the specified list of hosts.
        Syntax: reenable_hosts host1 host2 ...
        """
        for h in hosts:
            self.clusters[h] = _ALL_CLUSTERS[h]
                                                
    def complete_reenable_hosts(self, text, line, begidx, endidx):
        """Command line completion for reenable_hosts."""
        return [h for h in _ALL_CLUSTERS if h not in self.clusters]

    def do_show_clusters(self, line):
        """Print the list of clusters."""
        for c in self.clusters.values():
            print(c)

    def do_ping(self, line):
        """
        Ping the hosts, e.g. ping host1 host2
        """
        for cluster in self._select_clusters(line):
            print(cluster.ping())

    def do_all_jobs(self, line):
        """
        Report all the jobs in the queue. e.g all_jobs host1 host2
        """
        for cluster in self._select_clusters(line):
            print(cluster.get_all_jobs(prefix=True))
                                                                
    def do_user_jobs(self, line):
        """
        Report the jobs of the users. e.g user_jobs host1 host2
        """
        for cluster in self._select_clusters(line):
            print(cluster.get_user_jobs(prefix=True))

    def do_qinfo(self, line):
        """Report info on the queue manager."""
        for cluster in self._select_clusters(line):
            print(cluster.get_qinfo(prefix=True))

    #def do_qload(self, line):
    #    """info on the load of the clusters."""
    #    for cluster in self._select_clusters(line):
    #        print(cluster.get_qload())

    def do_run(self, line):
        """
        Execute a shell command on all hosts in the list.
        E.g. run uname -a host1 host2
        If the host list is empty, the command will be executed on all the hosts.
        """
        # split string into command and list of hostnames.
        tokens = line.split()
        for i, tok in enumerate(tokens):
            if tok in self.clusters:
                command = " ".join(tokens[:i])
                hostnames = " ".join(tokens[i:])
                break
        else:
            command, hostnames = s, None

        #print("command", command, "hosts", hostnames)
        for cluster in self._select_clusters(hostnames):
            result = cluster.ssh_exec(command)
            print(cluster.prefix_str(result.out))

    def do_abicheck(self, line):
        """Test if the abinit environment is properly setup on the remote hosts."""

        print("Testing abipy environment on the different clusters.")

        for cluster in self._select_clusters(line):
            print("%s ..." % cluster, end="")
            print(cluster.abicheck())
            #shell.exit()

    def do_abinit_version(self, line):
        """Show the abinit version available on the clusters."""
        for cluster in self._select_clusters(line):
            s = cluster.invoke_abinit("--version")
            print(cluster.prefix_str(s))

    def do_abinit_build_info(self, line):
        """Show the abinit build parameters."""
        for cluster in self._select_clusters(line):
            s = cluster.invoke_abinit("--build")
            print(cluster.prefix_str(s))

    def do_flow_start(self, line):
        """
        Start the flow on the remote cluster.
        Syntax: flow_start script.py hostname.
        """
        tokens = line.split()

        if len(tokens) != 2:
            print(self.do_flow_start.__doc__)
            return

        # Parse input line (local path of the script and hostname).
        script, hostname = tokens[0], tokens[1]

        # Start the flow on the remote host.
        self.flows_db.start_flow(script, hostname)

    def complete_flow_start(self, text, line, begidx, endidx):
        """Command line completion for flow_start."""
        tokens = line.split()

        if len(tokens) == 2:
            files = [f for f in os.listdir(os.getcwd()) if f.endswith(".py")]
            return [i for i in files if i.startswith(text)]

        elif len(tokens) == 3:
            return self.complete_hostnames(text, line, begidx, endidx)

        return []

    def do_show_flows(self, line):
        """Show the flows saved in the database."""
        print(self.flows_db)

    def do_flow_status(self, line):
        """
        Inspect the status of the flow on the remote cluster.
        Syntax: flow_status flow_workdir(s)
        """
        tokens = line.split()

        if not tokens:
            print(self.do_flow_status.__doc__)
            return

        for workdir in tokens:
            params = self.flows_db.get(workdir, None)
            if params is None: continue

            cluster = self.clusters[params.get("hostname")]

            shell = cluster.invoke_shell()
            cmd= "abirun.py %s status" % workdir
            print(cmd)
            result = shell.myexec(cmd)
            print(result)
            #return_code = shell.exit()
            #print("return_code", return_code)

    def complete_flow_status(self, text, line, begidx, endidx):
        """Command line completion for flow_status."""
        tokens = line.split()

        if len(tokens) == 1:
            return list(self.flows_db.keys())
                                                                              
        elif len(tokens) == 2:
            return [f for f in self.flows_db if f.startswith(text)]
                                                                              
        return []

    def do_flow_conf(self, line):
        """Show the YAML configuration files used on the remote hosts."""

        conf_files = [
            "~/.abinit/abipy/taskmanager.yml",
            "~/.abinit/abipy/scheduler.yml",
        ]

        for cluster in self._select_clusters(line):
            for s in self.yaml_configurations():
                print(cluster.prefix_str(s))


class FlowEntry(AttrDict):
    """
    An entry of the database with information on the `AbinitFlow` that is
    being executed on the remote host.

    .. attributes:

        hostname:
            Name of the remote host.
        workdir: 
            Absolute path of the working directory where the flow is being executed.
        start_date: 
            String with the creation date.
        status:
            Status of the flow.
    """
    #def __init__(self, *args, **kwargs):
    #    super(FlowEntry, self).__init__(*args, **kwargs)

    #    # Test the presence of mandatory keys.
    #    for key in ["hostname", "workdir", "start_date", "status"]:
    #        if key not in self:
    #            raise ValueError("Mandatory key %s is missing!" % key)

    #TODO: script?
    def __eq__(self, other):
        return (self.hostname == other.hostname and 
                self.workdir == other.workdir)

    def __ne__(self, other):
        return not self == other


class FlowsDatabase(collections.MutableMapping):
    """
    Database of flows executed on the clusters. It essentially consists of a dictionary that
    maps the name of the script used to generate the flow to the absolute path on the remote hosts
    where the flow is being executed.

    We use JSON to save/write the database to disk.

    .. Attributes:
        
        filepath:
            Absolute path of the JSON file.
    """
    #VERSION = "1"

    JSON_FILE = "flowsdb.json"
    # hostname --> [ {flow_workdir, start_date, status}, ... ]

    def __init__(self, filepath, db=None):
        self.filepath = os.path.abspath(filepath)
        self.db = {} if db is None else db

        # Add hostname to each entry and convert to AttrDict.
        if self.db:
            for hostname, entries in self.items():
                entries_with_hostname = []
                for e in entries:
                    if "hostname" not in e: e["hostname"] = hostname
                    entries_with_hostname.append(AttrDict(**e))

                self.db[hostname] = entries_with_hostname

        self.clusters = init_clusters()

    @classmethod
    def from_file(cls, filepath):
        """Read the database from file."""
        filepath = os.path.abspath(filepath)

        with open(filepath, "r") as fh:
            return cls(filepath, db=json.load(fh))

    @classmethod
    def from_user_config(cls):
        """
        Initialize the `FlowsDatabase` from the YAML file 'flowsdb.json'.
        Search first in the working directory and then in the configuration
        directory of abipy. If no JSON file, a new database is created in ~/.abinit/.abipy
        """
        # Try in the current directory.
        path = os.path.join(os.getcwd(), cls.JSON_FILE)

        if os.path.exists(path):
            return cls.from_file(path)

        # Try in the configuration directory.
        home = os.getenv("HOME")
        dirpath = os.path.join(home, ".abinit", "abipy")
        path = os.path.join(dirpath, cls.JSON_FILE)

        if os.path.exists(path):
            return cls.from_file(path)

        # Create empty database.
        return cls(path)

    def __str__(self):
        strio = StringIO.StringIO()
        pprint.pprint(self.db, stream=strio, indent=1)
        return "%s: %s\n" % (self.__class__.__name__, self.filepath) + strio.getvalue()

    # abc protocol.
    def __getitem__(self, key):
        return self.db[key]

    def __setitem__(self, key, value):
        self.db[key] = value

    def __delitem__(self, key):
        del self.db[key]

    def __iter__(self):
        return iter(self.db)

    def __len__(self):
        return len(self.db)
    # end abc protocol.

    def json_dump(self):
        """Dump the database in JSON format."""
        with open(self.filepath, "w") as fh:
            json.dump(self.db, fh, indent=4, sort_keys=4)

    def hostnames(self):
        """List of hostnames."""
        return list(self.keys())

    def _select_hosts(self, hostnames):
        """
        Helper function to select hostnames.
        Receives None, string or list of strings and returns a list of hostnames. 
        """
        if hostnames is None:
            return self.hostnames()
        else:
            return [hostnames] if is_string(hostnames) else hostnames

    def select_flows(self, hostname, status=None):
        if status is not None:
            return [flow for flow in self[hostname] if flow.status == status]
        else:
            return [flow for flow in self[hostname]]

    def flowdirs(self, hostname, status=None):
        if status is not None:
            return [flow.workdir for flow in self[hostname] if flow.status == status]
        else:
            return [flow.workdir for flow in self[hostname]]

    def has_flow(self, hostname, flowdir):
        """True if flowdir@hostname is in the database."""
        return flowdir in self.flowdirs(hostname)

    def register_flow(self, hostname, flowdir):
        """
        Add a new entry in the database.

        Args:
            hostname:
                Name of the remote host where the flow is being executed.
            flowdir:
                Absolute path on the remove cluste containing the `AbinitFlow`.
        """
        entries = self[hostname]

        new_entry = FlowEntry(hostname=hostname,
                              workdir=flowdir,
                              start_date=time.asctime(),
                              status="init")

        if new_entry in entries:
            raise ValueError("Entry %s is already in the database:\n %s" % (new_entry, entries))

        entries.append(new_entry)

        self.json_dump()

    def remove_flow(self, flow):
        """
        Remove a flow from the database.

        Returns:
            The entry that has been removed.
        """
        flows = self[flow.hostname]
        flows.remove(flow)

        self.json_dump()
        return flow

    def check_status(self, hostnames=None, workdir=None):
        """
        Check the status of the flows running on hostnames.

        Returns: 
            (results, changed)
            
            changed is the list of flows whose status has changed.
        """
        results, changed = [], []

        for host in self._select_hosts(hostnames):
            cluster = self.clusters[host]

            for flow in self[host]:
                if workdir is not None and flow.workdir != workdir:
                    continue
                
                shell = cluster.invoke_shell()
                cmd= "abirun.py %s status" % flow.workdir
                #print(cmd)
                result = shell.myexec(cmd)
                results.append(result)
                print(result)

                # TODO
                #if flow.status != new_status:
                #   flow.status = new_status
                #   flow.last_check = time.asctime()
                #   changed.append(flow)

        if changed:
            # Update the database.
            self.json.dump()

        return results, changed

    def maintenance(self, hostnames=None):
        """
        Remove dead flows from the database i.e. the flows whose workdir does not exist anymore.
        """
        removed = []

        for host in self._select_hosts(hostnames):
            cluster = clusters[host]
            for flow in self[host]:
                if not cluster.exists(flow.workdir):
                     removed.append(self.remove_flow(self, flow))

        return removed

    def start_flow(self, script, hostname):
        """
        Start the flow on the remote cluster.

        Args:
            script:
                Python script on the local machine
            hostname:
                Name of the remote host where the flow will be executed.
        """
        cluster = self.clusters[hostname]

        # Build absolute paths on the remote host: foo.py --> WORKDIR/foo/foo.py
        rdir_basename = os.path.basename(script).replace(".py", "")
        flow_rdir = os.path.join(cluster.workdir, rdir_basename)
        script_rpath = os.path.join(flow_rdir, os.path.basename(script))

        if self.has_flow(cluster.hostname, flow_rdir):
            raise ValueError("%s@%s is already in the database" % (flow_rdir, cluster.hostname))

        # Create directory if it does not exist.
        cluster.make_workdir()

        if cluster.exists(flow_rdir):
            raise ValueError("%s:%s already exists" % (cluster.hostname, flow_rdir))

        cluster.make_dir(flow_rdir)

        # Upload the script and make it executable.
        print("Uploading %s to %s:%s" % (script, hostname, script_rpath))
        sftp = cluster.sftp
        sftp.put(localpath=script, remotepath=script_rpath, confirm=True)
        sftp.chmod(script_rpath, mode=0700)
        #sftp.close()

        # Start a shell on the remote host and run the script to build the flow.
        command = "%s -w %s" %  (script_rpath, flow_rdir)
        shell = cluster.invoke_shell()
        result = shell.myexec(command)

        if result.return_code:
            print(result)
            print("%s returned %s. Aborting operation" % (command, result.return_code))
            return

        # Run the scheduler with nohup.
        # This is the most delicate part: not so sure that nohup will work everywhere.
        log_file = os.path.join(flow_rdir, "sched.log")
        #sched_cmd = "nohup abirun.py %s scheduler > /dev/null 2>&1 &" % flow_rdir
        sched_cmd = "nohup abirun.py %s scheduler > %s 2>&1 &" % (flow_rdir, log_file)
        print("Detaching process via: %s" % sched_cmd)
        shell = cluster.invoke_shell()
        shell.sendall(sched_cmd)
        return_code = shell.exit()
        print(return_code)
        #result = shell.myexec(sched_cmd)
        #print(result)

        if result.return_code:
            print("%s returned %s.\n Will remove the directory and abort" % (sched_cmd, result.return_code))
            #cluster.ssh.exec("rm -rf %s" % flow_absdir)
            #return

        # Add the flow to the local database.
        self.register_flow(cluster.hostname, flow_rdir)

    def cancel_flow(self, hostname, flow_dir):
        """
        Cancel the flow on the remote cluster.

        Args:
            hostname:
                Name of the remote host
            flow_dir:
                Directory on the remote host where the flow is being executed.
        """
        cluster = self.clusters[hostname]

        if not self.has_flow(hostname, flow_dir):
            raise ValueError("%s@%s is not in the database" % (flow_dir, hostname))

        # Call abirun to cancell the flow 
        command = "abirun.py %s cancel > /dev/null 2>&1 &" % flow_dir
        shell = cluster.invoke_shell()
        shell.sendall(command)
        return_code = shell.exit()
        print(return_code)

        #if result.return_code:
        #    print("%s returned %s.\n Will remove the directory and abort" % (command, result.return_code))
        #    #cluster.ssh.exec("rm -rf %s" % flow_absdir)
        #    #return

        ## Add the flow to the local database.
        #self.register_flow(cluster.hostname, flow_dir)

from abipy.core.testing import AbipyTest

class FlowsDatabaseTest(AbipyTest):

    def test_dbapi(self):

        db = {"manneback": [
                AttrDict(workdir="/WORKDIR/run_si_ebands", 
                         status="init", 
                         start_date="Wed Nov 20 19:59:05 2013"),
                AttrDict(workdir="/WORKDIR/run_si_ebands2", 
                         status="error", 
                         start_date="Wed Nov 21 19:59:05 2013"),
                ],

              "lemaitre2": [
                AttrDict(workdir="/WORKDIR/run_si_ebands", 
                         status="init", 
                         start_date="Wed Nov 20 19:59:05 2013"),
                ]
            }

        #self.tmpfile_write(string)
        fdb = FlowsDatabase(filepath="test.json", db=db) 

        print(fdb)

        # Cannot add the same flow twice.
        with self.assertRaises(ValueError):
            fdb.register_flow("manneback", "/WORKDIR/run_si_ebands")

        aequal = self.assertEqual
        atrue = self.assertTrue

        aequal(fdb.flowdirs("manneback", status="init"), ["/WORKDIR/run_si_ebands"])
        aequal(set(fdb.flowdirs("manneback")), set(["/WORKDIR/run_si_ebands", "/WORKDIR/run_si_ebands2"]))
        atrue(fdb.has_flow("manneback", "/WORKDIR/run_si_ebands2"))
        atrue( not fdb.has_flow("lemaitre2", "/WORKDIR/run_si_ebands2"))

        fdb.json_dump()
        other = FlowsDatabase.from_file(fdb.filepath)
        #assert 0


if __name__ == "__main__":
    import sys
    RunCommand().cmdloop()
    sys.exit(0)
