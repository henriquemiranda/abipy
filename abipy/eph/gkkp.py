"""
This file contains a series of plotting scripts 
for electron-phonon matrix elements
"""
import os
import re
import numpy as np
from collections import defaultdict
from abipy.iotools import ETSF_Reader
from monty.string import marquee

__all__ = [
    "GKKPPlotter",
    "GKKPReader",
    "MultiGKKPReader"
]

class GKKPPlotter():
    """
    Plot the electron-phonon matrix elements
    """
    def __init__(self,reader):
        pass

    def __str__(self):
        lines = []; app = lines.append
        return "\n".join(lines)

class GKKPReader():
    """
    Read a single GKKP file
    """
    def __init__(self,filename):
        self.filename = filename

        # each file corresponds to a q-point and perturbation
        with ETSF_Reader(filename) as db:
            # read the qpoint
            self.qpt = db.read_value('current_q_point')
            #read kptopt
            self.kptopt = db.read_value('kptopt')

            #TODO: add consistency check here
            #read number of kpoints
            self.nkpoints = db.read_dimvalue('number_of_kpoints')
            self.kpoints = db.read_value('reduced_coordinates_of_kpoints')
            self.nbands = db.read_dimvalue('max_number_of_states')
            self.natoms = db.read_dimvalue('number_of_atoms')
            self.nsppol = int(db.read_dimvalue('product_mband_nsppol')/self.nbands)
           
            #read the lattice 
            self.structure = db.read_structure()

        # read the perturbation
        basename = os.path.basename(filename)
        self.ipert = int(re.findall('([0-9]+)',basename)[0])-1

    def get_allk_gkkp(self,spin,ib1,ib2):
        """
        TODO: Should take slices of the netcdf file on the fly
        without reading all the contents
        """
        gkkp = self.get_all_gkkp(self)
        return gkkp[ib1,:,ib2,spin]

    def get_one_gkkp(self,ispin,ikpt,ib1,ib2):
        """
        TODO: Should take slices of the netcdf file on the fly
        without reading all the contents
        """
        gkkp = self.get_all_gkkp()
        return gkkp[ib1,ikpt,ib2,ispin]

    def get_all_gkkp(self):
        """
        Get gkkp matrix elements from the file
        """
        with ETSF_Reader(self.filename) as db:
            gkkp_reim = db.rootgrp.variables['second_derivative_eigenenergies_actif'][:]
            gkkp_reim = gkkp_reim.reshape(self.nbands,self.nkpoints,self.nbands,self.nsppol,2)
            gkkp = gkkp_reim[:,:,:,:,0] + 1j*gkkp_reim[:,:,:,:,1]
        return gkkp

    def __str__(self):
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__, mark="=")) 
        app('qpoint: '+("%12.8lf "*3)%tuple(self.qpt))
        app('ipert:  %d'%self.ipert)
        app('nbands: %d'%self.nbands)
        app('natoms: %d'%self.natoms)
        app('nkpoints: %d'%self.natoms)
        return "\n".join(lines)

class MultiGKKPReader():
    """
    Read the electron-phonon matrix elements from eph_task 5
    In the init we only read the basic informations of the different gkkp files.
    The matrix elements are only loaded uppon request.

    The reader should be able to read from different file formats on disk
    and provide a common interface to access the data.
    Currently works with GKKP files produced with eph_task 2
    """
    def __init__(self,filenames_list,ddb_file):

        self.ddb_file = ddb_file
        self.filenames_list = filenames_list
        self._qpt_ipert_dict = defaultdict(dict)

        for filename in filenames_list:
          
            #read data from this GKKP file 
            gkkp = GKKPReader(filename)
 
            #store qpoint and perturbation
            qpt = tuple(gkkp.qpt)
            ipert = gkkp.ipert
            self._qpt_ipert_dict[qpt][ipert] = filename

            #store the structure
            self.structure = gkkp.structure

        self.kpoints = gkkp.kpoints
        self.qpoints = list(self._qpt_ipert_dict.keys())
        self.nbands = gkkp.nbands
        self.natoms = gkkp.natoms
        self.nmodes = self.natoms*3

    def _get_phonons(self):
        """Call anaddb to diagonalize the dynamical matrices in the DDB file"""
        from abipy import abilab
        ddb = abilab.abiopen(self.ddb_file)
        phonon_modes = []
        phonon_freqs = []
        for qpoint in self.qpoints:
            phbands = ddb.anaget_phmodes_at_qpoint(qpoint)
            phonon_modes.append(phbands.phdispl_cart)
            phonon_freqs.append(phbands.phfreqs)
        phonon_freqs = np.vstack(phonon_freqs)
        phonon_modes = np.vstack(phonon_modes)
        return phonon_freqs, phonon_modes

    @property
    def phonon_modes(self):
        if not hasattr(self,'_phonon_modes'):
            self._phonon_freqs, self._phonon_modes = self._get_phonons()
        return self._phonon_modes

    @property
    def phonon_freqs(self):
        if not hasattr(self,'_phonon_freqs'):
            self._phonon_freqs, self._phonon_modes = self._get_phonons()
        return self._phonon_freqs

    @property
    def nkpoints(self):
        return len(self.kpoints)

    @property
    def nqpoints(self):
        return len(self.qpoints)

    @classmethod
    def from_flowdir(cls,flowdir):
        """Find all GKKP files in the flow"""
        flow = flowtk.Flow.pickle_load(flowdir)
        return cls.from_flow(flowdir)

    @classmethod
    def from_flow(cls,flow):
        """Find all the GKKP files in this flow and a DDB file"""
        filenames_list = []
        ddb_file = None
        for work in flow:
            for task in work:
                filenames_list.extend(task.outdir.has_abiext("GKK",single_file=False))
                has_ddb_file = task.indir.has_abiext("DDB")
                if has_ddb_file: ddb_file = has_ddb_file
        if ddb_file is None: raise FileNotFoundError('DDB file not found in flow')
        return cls(filenames_list,ddb_file)

    @classmethod
    def from_work(cls,flowdir):
        """Find all GKKP files in this work and a DDB file"""
        filenames_list = []
        ddb_file = None
        for task in work:
            filenames_list.extend(task.outdir.has_abiext("GKK",single_file=False))
            has_ddb_file = task.indir.has_abiext("DDB")
            if has_ddb_file: ddb_file = has_ddb_file
        if ddb_file is None: raise FileNotFoundError('DDB file not found in flow')
        return cls(filenames_list,ddb_file)

    def get_gkkp_fix_k(self,ispin,kpt,ib1,ib2,imu,cartesian=True):
        """
        Get a slice of GKKP matrix elements by fixing k
        the resulting array has dimensions of nqpoints
        """
        ikpt = self.get_ikpt(kpt)
        gkkp_fix_k = np.zeros([self.nqpoints],dtype=complex)
        for iqpt,qpt in enumerate(self.qpoints):
            gkkp_fix_k[iqpt] = self.get_one_gkkp(iqpt,ispin,ikpt,ib1,ib2,imu,cartesian=cartesian)
        return self.qpoints, gkkp_fix_k

    def get_gkkp_fix_q(self,ispin,qpt,ib1,ib2,imu,cartesian=True):
        """
        Get a slice of GKKP matrix elements by fixing q
        the resulting array has dimensions of nkpoints
        """
        iqpt = self.get_ikpt(qpt)
        gkkp_fix_q = np.zeros([self.nkpoints],dtype=complex)
        for ikpt,kpt in enumerate(self.kpoints):
            gkkp_fix_q[ikpt] = self.get_one_gkkp(iqpt,ispin,ikpt,ib1,ib2,imu,cartesian=cartesian)
        return self.kpoints, gkkp_fix_q

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
        gkkp_rec = []
        for i in range(self.nmodes):
            reader = self.get_reader(iqpt,i)
            gkkp = reader.get_one_gkkp(ispin,ikpt,ib1,ib2)
            gkkp_rec.append(gkkp)

        gkkp_rec = np.array(gkkp_rec)

        #project cartesian
        gkkp_cartesian = np.einsum('a,ma->m', gkkp_rec, self.tcartesian)
        if cartesian: return gkkp_cartesian[imu]

        #project phonon modes 
        t = self.phonon_modes[iqpt]
        gkkp_modes = np.einsum('a,ma->m', gkkp_cartesian, t)
        return gkkp_modes[imu]

    def get_filename(self,iqpt,ipert):
        """
        Get the filename containing the information of this qpoint and perturbation
        """
        qpt = self.qpoints[iqpt]
        return self._qpt_ipert_dict[qpt][ipert]
 
    def get_reader(self,iqpt,ipert):
        """
        Get reader for a particular qpoint and perturbation
        an ETSF_Reader is returned, the user is responsible for closing
        """   
        filename = self.get_filename(iqpt,ipert)
        return GKKPReader(filename)

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
            app(("%12.8lf "*3)%qpt)
        return "\n".join(lines)

