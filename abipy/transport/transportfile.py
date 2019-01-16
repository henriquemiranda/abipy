# coding: utf-8
"""TRANSPORT.nc file."""

import numpy as np
import pymatgen.core.units as units
import abipy.core.abinit_units as abu

from monty.functools import lazy_property
from abipy.core.mixins import AbinitNcFile, Has_Header, Has_Structure, Has_ElectronBands, NotebookWriter
from abipy.electrons.ebands import ElectronsReader
from .result import TransportResult, TransportResultRobot

__all__ = [
    "TransportFile",
]

class TransportFile(AbinitNcFile, Has_Header, Has_Structure, Has_ElectronBands):
    @classmethod
    def from_file(cls, filepath):
        """Initialize the object from a netcdf_ file."""
        return cls(filepath)

    def __init__(self, filepath):
        super(TransportFile, self).__init__(filepath)
        self.reader = TransportReader(filepath)

        ebands = self.reader.read_ebands()
        self.fermi = ebands.fermie*abu.eV_Ha
        self.volume = ebands.structure.volume*abu.Ang_Bohr**3

        self.tmesh = self.reader.tmesh

    @property
    def ntemp(self):
        return len(self.tmesh)

    @lazy_property
    def ebands(self):
        """|ElectronBands| object."""
        return self.reader.read_ebands()

    @property
    def structure(self):
        """|Structure| object."""
        return self.ebands.structure

    @lazy_property
    def params(self):
        """:class:`OrderedDict` with parameters that might be subject to convergence studies."""
        od = self.get_ebands_params()
        return od

    def get_boltztrap_result(self,itemp=None,tmesh=None):
        """
        Get one instance of TransportResult according to itemp

        Args:
            itemp: the index of the temperature from which to create the TransportResult class
        """
        reader = self.reader
        wmesh, dos, idos = reader.read_dos()

        if itemp is None:
            wmesh, vvdos = reader.read_vvdos()
            tau_temp = None
        else:
            wmesh, vvdos_tau = reader.read_vvdos_tau()
            vvdos = vvdos_tau[itemp]
            tau_temp = reader.tmesh[itemp]

        if tmesh is None: tmesh = [ 50*i for i in range(1,11)]

        #todo spin
        #self.nsppol
        dos = dos[0]
        vvdos = vvdos[:,:,0]
        return TransportResult(wmesh,dos,vvdos,self.fermi,tmesh,self.volume,self.nelect,tau_temp=tau_temp,nsppol=1,margin=0.1)

    def get_boltztrap_results(self,tmesh=None):
        """
        Return multiple instances of TransportResults from the data in the TRANSPORT.nc file
        """
        results = [self.get_boltztrap_result(tmesh=tmesh,itemp=itemp) for itemp in [None]+list(range(self.ntemp))]
        return TransportResultRobot(results)

    def close(self):
        """Close the file."""
        self.reader.close()

class TransportReader(ElectronsReader):
    """
    This class reads the results stored in the TRANSPORT.nc file
    It provides helper function to access the most important quantities.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        super(TransportReader, self).__init__(filepath)
        ktmesh = self.read_value("kTmesh")
        self.tmesh = ktmesh / abu.kb_HaK

    def read_vvdos(self):
        """
        Read the group velocity density of states
        The vvdos_vals array has 3 dimensions (9,nsppolplus1,nw)
          1. 3x3 components of the tensor
          2. the spin polarization + 1 for the sum
          3. the number of frequencies
        """
        vals = self.read_variable("vvdos_mesh")
        wmesh = vals[:]
        vals = self.read_variable("vvdos_vals")
        nsppol = vals.shape[1]-1
        vvdos = vals[:,:,1:,:]
        return wmesh, vvdos

    def read_vvdos_tau(self):
        """
        Read the group velocity density of states times lifetime for different temperatures
        The vvdos_tau array has 4 dimensions (ntemp,9,nsppolplus1,nw)
          1. the number of temperatures
          2. 3x3 components of the tensor
          3. the spin polarization + 1 for the sum
          4. the number of frequencies
        """
        vals = self.read_variable("vvdos_mesh")
        wmesh = vals[:]
        vals = self.read_variable("vvdos_tau")
        nsppol = vals.shape[1]-1
        vvdos_tau = vals[:,:,:,1:,:]
        return wmesh, vvdos_tau

    def read_dos(self):
        """
        Read the density of states
        """
        vals = self.read_variable("edos_mesh")
        wmesh = vals[:]
        vals = self.read_variable("edos_dos")
        dos = vals[1:,:]
        vals = self.read_variable("edos_idos")
        idos = vals[1:,:]
        return wmesh, dos, idos

    def read_evk_diagonal(self):
        """
        Read the group velocities i.e the diagonal matrix elements.
        Return (nsppol, nkpt) |numpy-array| of real numbers.
        """
        vels = self.read_variable("vred_diagonal")
        # Cartesian? Ha --> eV?
        return vels * (units.Ha_to_eV / units.bohr_to_ang)

    def read_evk_skbb(self):
        return self.read_value("h1_matrix", cplx_mode="cplx") * (units.Ha_to_eV / units.bohr_to_ang)
