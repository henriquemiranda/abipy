# coding: utf-8
"""TRANSPORT.nc file."""

import numpy as np
import pymatgen.core.units as units
import abipy.core.abinit_units as abu

from monty.functools import lazy_property
from abipy.core.mixins import AbinitNcFile, Has_Header, Has_Structure, Has_ElectronBands, NotebookWriter
from abipy.transport import TransportResult, TransportResultRobot

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
        Get one instance of Boltztrap result according to itemp

        Args:
            itemp: the index of the temperature from which to create the TransportResult class
        """
        reader = self.reader
        wmesh, dos, idos = reader.read_dos()

        if itemp is None:
            wmesh, vvdos = reader.read_vvdos()
            tau_temp = None
