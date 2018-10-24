"""
This file contains a series of plotting scripts 
for electron-phonon matrix elements
"""
import os
import re
import numpy as np
from collections import defaultdict
from abipy.electrons.ebands import ElectronsReader
from monty.string import marquee
from abipy.tools.plotting import add_fig_kwargs

__all__ = [
    "GKKPReader",
    "MultiGKKPReader"
]

class GKKPReader(ElectronsReader):
    """
    Read a single GKKP file
    """
    def __init__(self,filename):
        super(GKKPReader, self).__init__(filename)

        # read the qpoint
        self.qpt = self.read_value('qpoint')

        #read kptopt
        self.kptopt = self.read_value('kptopt')

        #TODO: add consistency check here
        #read number of kpoints
        self.nkpoints = self.read_dimvalue('number_of_kpoints')
        self.kpoints = self.read_value('reduced_coordinates_of_kpoints')
        self.nbands = self.read_dimvalue('max_number_of_states')
        self.natoms = self.read_dimvalue('number_of_atoms')
        self.nsppol = self.read_dimvalue('number_of_spins')
        
        # read the perturbation
        basename = os.path.basename(filename)

    @property
    def ebands(self):
        """read ebands"""
        return self.read_ebands()

    @property
    def structure(self):
        """read the lattice"""
        return self.read_structure()
     
    def get_allk_gkkp(self,ispin,imu,ib1,ib2):
        """
        TODO: Should take slices of the netcdf file on the fly
        without reading all the contents
        """
        gkkp = self.get_all_gkkp()
        return gkkp[ispin,:,imu,ib1,ib2]

    def get_allm_gkkp(self,ispin,ikpt,ib1,ib2):
        """
        TODO: Should take slices of the netcdf file on the fly
        without reading all the contents
        """
        gkkp = self.get_all_gkkp()
        return gkkp[ispin,ikpt,:,ib1,ib2]

    def get_one_gkkp(self,ispin,ikpt,imu,ib1,ib2):
        """
        TODO: Should take slices of the netcdf file on the fly
        without reading all the contents
        """
        gkkp = self.get_all_gkkp()
        return gkkp[ispin,ikpt,imu,ib1,ib2]

    def get_all_gkkp(self):
        """
        Get gkkp matrix elements from the file
        """
        gkkp_reim = self.rootgrp.variables['gkq'][:]
        gkkp = gkkp_reim[:,:,:,:,:,0] + 1j*gkkp_reim[:,:,:,:,:,1]
        return gkkp

    def __str__(self):
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__, mark="=")) 
        app('qpoint: '+("%12.8lf "*3)%tuple(self.qpt))
        app('nbands: %d'%self.nbands)
        app('natoms: %d'%self.natoms)
        app('nkpoints: %d'%self.nkpoints)
        return "\n".join(lines)

class MultiGKKPReader():
    """
    Read the electron-phonon matrix elements from eph_task 2
    In the init we only read the basic informations of the different gkkp files.
    The matrix elements are only loaded uppon request.

    The reader should be able to read from different file formats on disk
    and provide a common interface to access the data.
    Currently works with GKKP files produced with eph_task 2
    """
    def __init__(self,filenames_list):

        self.filenames_list = filenames_list

        self.qpoints = []
        self.gkqs = []
        for filename in filenames_list:

            #open the reader
            gkq = GKKPReader(filename)
            self.gkqs.append(gkq)

            #store qpoint
            qpt = gkq.qpt
            self.qpoints.append(qpt)

            if np.linalg.norm(qpt) < 1e-8:
                self.kpoints = gkq.kpoints
                self.nbands = gkq.nbands
                self.natoms = gkq.natoms
                self.nmodes = self.natoms*3

            #store the structure
            self.structure = gkq.structure


    def __getitem__(self,iqpt):
        return self.gkqs[iqpt]

    @property
    def nkpoints(self):
        return len(self.kpoints)

    @property
    def nqpoints(self):
        return len(self.qpoints)

    @property
    def qpoints_dist(self):
        if not hasattr(self,'_qpoints_dist'):
            self._qpoints_dist = np.array([ np.linalg.norm(qpt) for qpt in self.qpoints ])
        return self._qpoints_dist

    @property
    def kpoints_dist(self):
        if not hasattr(self,'_kpoints_dist'): 
            self._kpoints_dist = np.array([ np.linalg.norm(kpt) for kpt in self.kpoints ])
        return self._kpoints_dist

    @classmethod
    def from_flowdir(cls,flowdir):
        """Find all GKKP files in the flow"""
        flow = flowtk.Flow.pickle_load(flowdir)
        return cls.from_flow(flowdir)

    @classmethod
    def from_flow(cls,flow):
        """Find all the GKQ files in this flow"""
        filenames_list = []
        ddb_file = None
        for work in flow:
            for task in work:
                filenames_list.extend(task.outdir.has_abiext("GKQ",single_file=False))
        return cls(filenames_list)

    @classmethod
    def from_work(cls,flowdir):
        """Find all GKQ files in this work"""
        filenames_list = []
        ddb_file = None
        for task in work:
            filenames_list.extend(task.outdir.has_abiext("GKQ",single_file=False))
        return cls(filenames_list)

    def get_gkkp_fix_k(self,ispin,kpt,imu,ib1,ib2,cartesian=True):
        """
        Get a slice of GKKP matrix elements by fixing k
        the resulting array has dimensions of nqpoints
        """
        ikpt = self.get_ikpt(kpt)
        gkkp_fix_k = np.zeros([self.nqpoints],dtype=complex)
        for iqpt,qpt in enumerate(self.qpoints):
            gkkp_fix_k[iqpt] = self.get_one_gkkp(iqpt,ispin,ikpt,ib1,ib2,imu,cartesian=cartesian)
        return self.qpoints, gkkp_fix_k

    def get_gkkp_fix_q(self,ispin,qpt,imu,ib1,ib2,cartesian=True):
        """
        Get a slice of GKKP matrix elements by fixing q
        the resulting array has dimensions of nkpoints
        """
        iqpt = self.get_ikpt(qpt)
        gkkp_fix_q = np.zeros([self.nkpoints],dtype=complex)
        for ikpt,kpt in enumerate(self.kpoints):
            gkkp_fix_q[ikpt] = self.get_one_gkkp(iqpt,ispin,ikpt,ib1,ib2,imu,cartesian=cartesian)
        return self.kpoints, gkkp_fix_q

    def plot_dispersionq_ax(self,ax,ispin,imu,ib1,ib2,kpt=(0,0,0),f=np.abs):
        """
        Plot the magnitude of the electron-phonon matrix elments along a path
        """
        kpoints, gkkp = self.get_gkkp_fix_k(ispin,kpt,imu,ib1,ib2)
        ax.scatter(self.qpoints_dist,f(gkkp))

    def plot_dispersionk_ax(self,ax,ispin,imu,ib1,ib2,qpt=(0,0,0),f=np.abs):
        """
        Plot the magnitude of the electron-phonon matrix elements along a path
        """
        qpoints, gkkp = self.get_gkkp_fix_q(ispin,qpt,imu,ib1,ib2)
        ax.scatter(self.kpoints_dist,f(gkkp))

    @add_fig_kwargs
    def plot_dispersionk(self,ispin,imu,ib1,ib2,qpt=(0,0,0),**kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_dispersionk_ax(ax,ispin,imu,ib1,ib2,qpt=qpt,**kwargs)
        return fig

    @add_fig_kwargs
    def plot_dispersionq(self,ispin,imu,ib1,ib2,kpt=(0,0,0),**kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        self.plot_dispersionq_ax(ax,ispin,imu,ib1,ib2,kpt=kpt,**kwargs)
        return fig

    @property
    def tcartesian(self):
        """ create tranformation matrix from reciprocal coordinates to cartesian """
        if not hasattr(self,"_tcartesian"):
            self._tcartesian = np.zeros([self.nmodes,self.nmodes])
            for a in range(self.natoms):
                self._tcartesian[a*3:(a+1)*3,a*3:(a+1)*3] = self.structure.reciprocal_lattice.matrix.T
        return self._tcartesian

    def get_one_gkkp(self,iqpt,ispin,ikpt,ib1,ib2,imu,cartesian=True):
        """
        Get one matrix element at a time using the indexes of the q and k point
        """
        gkkp_rec = self[iqpt].get_allm_gkkp(ispin,ikpt,ib1,ib2)

        #project cartesian
        gkkp_cartesian = np.einsum('a,ma->m', gkkp_rec, self.tcartesian)
        if cartesian: return gkkp_cartesian[imu]

        #project phonon modes 
        t = self.phonon_modes[iqpt]
        gkkp_modes = np.einsum('a,ma->m', gkkp_cartesian, t)
        return gkkp_modes[imu]

    def get_iqpt(self,qpt):
        """
        Find the index of a certain q-point
        """
        # This is a very naive search algorithm
        # TODO improve using a search tree or similar
        for iqpt,file_qpt in enumerate(self.qpoints):
            if np.isclose(file_qpt,qpt).all(): return iqpt
        return None

    def get_ikpt(self,kpt):
        """
        Find the index of a certain q-point
        """
        # This is a very naive search algorithm
        # TODO improve using a search tree or similar
        for ikpt,file_kpt in enumerate(self.kpoints):
            if np.isclose(file_kpt,kpt).all(): return ikpt
        return None

    def __str__(self):
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__, mark="=")) 
        app('nbands: %d'%self.nbands)
        app('natoms: %d'%self.natoms)
        app('kpoints:')
        for kpt in self.kpoints:
            app(("%12.8lf "*3)%tuple(kpt))
        app('qpoints:')
        for qpt in self.qpoints:
            app(("%12.8lf "*3)%tuple(qpt))
        return "\n".join(lines)

