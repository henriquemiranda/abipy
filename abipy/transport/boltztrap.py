# coding: utf-8
"""
This module containes a Bolztrap2 class to interpolate and analyse the results
It also provides interfaces with Abipy objects allowing to
initialize the Boltztrap2 calculation from Abinit files

Warning:

    Work in progress
"""
import pickle
import numpy as np
import pandas as pd
import abipy.core.abinit_units as abu
from itertools import product
from collections import OrderedDict

from monty.string import marquee
from monty.termcolor import cprint
from monty.dev import deprecated
from abipy.tools import duck
from abipy.electrons.ebands import ElectronBands
from abipy.core.kpoints import Kpath
from abipy.core.structure import Structure
from abipy.tools.decorators import timeit
from .result import TransportResult, TransportResultRobot

__all__ = [
    "AbipyBoltztrap",
]

class AbipyBoltztrap():
    """
    Wrapper to Boltztrap2 interpolator
    This class contains the same quantities as the Loader classes from dft.py in Boltztrap2
    Additionally it has methods to call the Boltztrap2 interpolator.
    It creates multiple instances of BolztrapResult storing the results of the interpolation
    Enter with quantities in the IBZ and interpolate to a fine BZ mesh
    """
    def __init__(self,fermi,structure,nelect,kpoints,eig,volume,linewidths=None,tmesh=None,
                 mommat=None,magmom=None,lpratio=5):
        #data needed by boltztrap
        self.fermi = fermi
        self.atoms = structure.to_ase_atoms()
        self.nelect = nelect
        self.kpoints = np.array(kpoints)
        self.volume = volume
        self.mommat = mommat
        self.magmom = magmom

        #additional parameters
        self.eig = eig
        self.structure = structure
        self.linewidths = linewidths
        self.tmesh = tmesh
        self.lpratio = lpratio

    @property
    def nkpoints(self):
        return len(self.kpoints)

    @property
    def equivalences(self):
        if not hasattr(self,'_equivalences'):
            self.compute_equivalences()
        return self._equivalences

    @property
    def coefficients(self):
        if not hasattr(self,'_coefficients'):
            self.compute_coefficients()
        return self._coefficients

    @property
    def linewidth_coefficients(self):
        if not hasattr(self,'_linewidth_coefficients'):
            self.compute_coefficients()
        return self._linewidth_coefficients

    @property
    def rmesh(self):
        if not hasattr(self,'_rmesh'):
            self.get_interpolation_mesh()
        return self._rmesh

    @property
    def nequivalences(self):
        return len(self.equivalences)

    @property
    def ncoefficients(self):
        return len(self.coefficients)

    @property
    def ntemps(self):
        return len(self.linewidths)

    def pickle(self,filename):
        with open(filename,'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def from_pickle(cls,filename):
        with open(filename,'rb') as f:
            cls = pickle.load(f)
        return cls

    @classmethod
    def from_ebands(cls):
        """Initialize from an ebands object"""
        raise NotImplementedError('TODO')

    @classmethod
    def from_evk(cls):
        """Intialize from a EVK file"""
        raise NotImplementedError('TODO')

    @classmethod
    def from_dftdata(cls,dftdata,el_temp=300,lpratio=5):
        """
        Initialize an instance of this class from a DFTData instance from Boltztrap

        Args:
            dftdata: DFTData
            el_temp: electronic temperature (in K) to use in the integrations with the Fermi-Dirac occupations
            lpratio: ratio to multiply by the number of k-points in the IBZ and give the
                     number of real space points inside a sphere
        """
        structure = Structure.from_ase_atoms(dftdata.atoms)
        return cls(dftdata.fermi,structure,dftdata.nelect,dftdata.kpoints,dftdata.ebands,
                   dftdata.get_volume(),linewidths=None,tmesh=None,
                   mommat=dftdata.mommat,magmom=None,lpratio=lpratio)

    @classmethod
    def from_sigeph(cls, sigeph, itemp_list=None, bstart=None, bstop=None, el_temp=300, lpratio=5):
        """
        Initialize interpolation of the bands and lifetimes from a SigEphFile object

        Args:
            sigeph: |SigEphFile| instance
            itemp_list: list of the temperature indexes to consider
            bstart, bstop: only consider bands between bstart and bstop
            el_temp: electronic temperature (in K) to use in the integrations with the Fermi-Dirac occupations
            lpratio: ratio to multiply by the number of k-points in the IBZ and give the
                     number of real space points inside a sphere
        """
        #get the lifetimes as an array
        qpes = sigeph.get_qp_array(mode='ks+lifetimes')

        #get other dimensions
        if bstart is None: bstart = sigeph.reader.max_bstart
        if bstop is None:  bstop  = sigeph.reader.min_bstop
        fermi  = sigeph.ebands.fermie*abu.eV_Ha
        structure = sigeph.ebands.structure
        volume = sigeph.ebands.structure.volume*abu.Ang_Bohr**3
        nelect = sigeph.ebands.nelect
        kpoints = [k.frac_coords for k in sigeph.sigma_kpoints]

        if sigeph.nsppol == 2:
            raise NotImplementedError("nsppol 2 not implemented")

        #TODO handle spin
        eig = qpes[0,:,bstart:bstop,0].real.T*abu.eV_Ha

        itemp_list = list(range(sigeph.ntemp)) if itemp_list is None else duck.list_ints(itemp_list)
        linewidths = []
        tmesh = []
        for itemp in itemp_list:
            tmesh.append(sigeph.tmesh[itemp])
            fermi = sigeph.mu_e[itemp]*abu.eV_Ha
            #TODO handle spin
            linewidth = qpes[0,:,bstart:bstop,itemp].imag.T*abu.eV_Ha
            linewidths.append(linewidth)

        return cls(fermi, structure, nelect, kpoints, eig, volume, linewidths=linewidths,
                   tmesh=tmesh, lpratio=lpratio)

    def get_lattvec(self):
        """this method is required by Bolztrap"""
        return self.lattvec

    @property
    def nbands(self):
        nbands, rpoints = self.coefficients.shape
        return nbands

    @property
    def lattvec(self):
        if not hasattr(self,"_lattvec"):
            self._lattvec = self.atoms.get_cell().T / abu.Bohr_Ang
        return self._lattvec

    def get_ebands(self,kpath=None,line_density=20,vertices_names=None,linewidth_itemp=False):
        """
        Compute the band-structure using the computed coefficients

        Args:
            kpath: |Kpath| instance where to interpolate the eigenvalues and linewidths
            line_density: Number of points used to sample the smallest segment of the path
            vertices_names:  List of tuple, each tuple is of the form (kfrac_coords, kname) where
                kfrac_coords are the reduced coordinates of the k-point and kname is a string with the name of
                the k-point. Each point represents a vertex of the k-path.
            linewith_itemp: list of indexes refering to the temperatures where the linewidth will be interpolated
        """
        from BoltzTraP2 import fite

        if kpath is None:
            if vertices_names is None:
                vertices_names = [(k.frac_coords, k.name) for k in self.structure.hsym_kpoints]

            kpath = Kpath.from_vertices_and_names(self.structure, vertices_names, line_density=line_density)

        #call boltztrap to interpolate
        coeffs = self.coefficients
        eigens_kpath, vvband = fite.getBands(kpath.frac_coords, self.equivalences, self.lattvec, coeffs)

        linewidths_kpath = None
        if linewidth_itemp is not False:
            coeffs = self.linewidth_coefficients[linewidth_itemp]
            linewidths_kpath, vvband = fite.getBands(kpath.frac_coords, self.equivalences, self.lattvec, coeffs)
            linewidths_kpath = linewidths_kpath.T[np.newaxis,:,:]*abu.Ha_eV

        #convert units and shape
        eigens_kpath   = eigens_kpath.T[np.newaxis,:,:]*abu.Ha_eV
        occfacts_kpath = np.zeros_like(eigens_kpath)
        nspinor1 = 1
        nspden1 = 1

        #return a ebands object
        return ElectronBands(self.structure, kpath, eigens_kpath, self.fermi*abu.Ha_eV, occfacts_kpath,
                             self.nelect, nspinor1, nspden1, linewidths=linewidths_kpath)

    @deprecated(message="get_bands is deprecated, use get_ebands")
    def get_bands(self, **kwargs):
        return self.get_ebands(**kwargs)

    def get_interpolation_mesh(self):
        """From the array of equivalences determine the mesh that was used"""
        max1, max2, max3 = 0,0,0
        for equiv in self.equivalences:
            max1 = max(np.max(equiv[:,0]),max1)
            max2 = max(np.max(equiv[:,1]),max2)
            max3 = max(np.max(equiv[:,2]),max3)
        self._rmesh = (2*max1+1,2*max2+1,2*max3+1)
        return self._rmesh

    def dump_rsphere(self,filename):
        """ Write a file with the real space points"""
        with open(filename, 'wt') as f:
            for iband in range(self.nbands):
                for ie,equivalence in enumerate(self.equivalences):
                    coeff = self.coefficients[iband,ie]
                    for ip,point in enumerate(equivalence):
                        f.write("%5d %5d %5d "%tuple(point)+"%lf\n"%((abs(coeff))**(1./3)))
                f.write("\n\n")

    @timeit
    def compute_equivalences(self):
        """Compute equivalent k-points"""
        from BoltzTraP2 import sphere
        if duck.is_listlike(self.lpratio): nkpt = self.lpratio
        else: nkpt = self.lpratio*self.nkpoints
        try:
            self._equivalences = sphere.get_equivalences(self.atoms, self.magmom, nkpt)
        except TypeError:
            self._equivalences = sphere.get_equivalences(self.atoms, nkpt)


    @timeit
    def compute_coefficients(self):
        """Call fitde3D routine from Boltztrap2"""
        from BoltzTraP2 import fite
        #we will set ebands to compute teh coefficients
        self.ebands = self.eig
        self._coefficients = fite.fitde3D(self, self.equivalences)

        if self.linewidths:
            self._linewidth_coefficients = []
            for itemp in range(self.ntemps):
                self.ebands = self.linewidths[itemp]
                coeffs = fite.fitde3D(self, self.equivalences)
                self._linewidth_coefficients.append(coeffs)

        #at the end we always unset ebands
        delattr(self,"ebands")

    @timeit
    def run(self,npts=500,dos_method='gaussian:0.05 eV',erange=None,el_temp=300,margin=0.1,nworkers=1,verbose=0):
        """
        Interpolate the eingenvalues to compute dos and vvdos
        This part is quite memory intensive

        Args:
            npts: number of frequency points
            dos_method: when using a patched version of Boltztrap
            el_temp: electronic temperature (in K) to use in the integrations with the Fermi-Dirac occupations
        """
        boltztrap_results = []; app = boltztrap_results.append
        import inspect
        from BoltzTraP2 import fite
        import BoltzTraP2.bandlib as BL

        def BTPDOS(eband,vvband,cband=None,erange=None,npts=None,scattering_model="uniform_tau",mode=dos_method):
            """
            This is a small wrapper for Boltztrap2 to use the official version or a modified
            verison using gaussian or lorentzian smearing
            """
            temp = None
            #in case of histogram mode we read the smearing here
            if "histogram" in mode:
                i = mode.find(":")
                if i != -1:
                    value, eunit = mode[i+1:].split()
                    if eunit == "eV": temp = float(value)*abu.eV_to_K
                    elif eunit == "Ha": temp = float(value)*abu.Ha_to_K
                    elif eunit == "K": temp = float(value)
                    else: raise ValueError('Unknown unit %s'%eunit)
                    mode = mode.split(':')[0]

            try:
                wmesh, dos_tau, vvdos_tau, _ = BL.BTPDOS(eband, vvband, erange=erange, npts=npts,
                                                         scattering_model=scattering_model, mode=dos_method)
            except TypeError:
                if 'histogram' not in mode:
                    print('Could not pass \'dos_method=%s\' argument to Bolztrap2. '
                          'Falling back to histogram method'%dos_method)
                wmesh, dos_tau, vvdos_tau, _ = BL.BTPDOS(eband, vvband, erange=erange, npts=npts, scattering_model=scattering_model)

            if 'histogram' in mode and temp:
                dos_tau = BL.smoothen_DOS(wmesh,dos_tau,temp)
                for i,j in product(range(3),repeat=2):
                    vvdos_tau[i,j] = BL.smoothen_DOS(wmesh,vvdos_tau[i,j],temp)

            return wmesh, dos_tau, vvdos_tau

        #TODO change this!
        if erange is None: erange = (np.min(self.eig)-margin,np.max(self.eig)+margin)
        else: erange = np.array(erange)/abu.Ha_eV+self.fermi

        #interpolate the electronic structure
        if verbose: print('interpolating bands')
        results = fite.getBTPbands(self.equivalences, self.coefficients,
                                   self.lattvec, nworkers=nworkers)
        eig_fine, vvband, cband = results

        #calculate DOS and VDOS without lifetimes
        if verbose: print('calculating dos and vvdos without lifetimes')
        wmesh,dos,vvdos = BTPDOS(eig_fine, vvband, erange=erange, npts=npts, mode=dos_method)
        app(TransportResult(wmesh,dos,vvdos,self.fermi,el_temp,self.volume,self.nelect,margin=margin))

        #if we have linewidths
        if self.linewidths:
            for itemp in range(self.ntemps):
                if verbose: print('itemp %d\ninterpolating bands')
                #calculate the lifetimes on the fine grid
                results = fite.getBTPbands(self.equivalences, self._linewidth_coefficients[itemp],
                                           self.lattvec, nworkers=nworkers)
                linewidth_fine, vvband_, cband_ = results
                tau_fine = 1.0/np.abs(2*linewidth_fine*abu.Ha_s) # NOTE conversion from eV to Hartree done before

                #calculate vvdos with the lifetimes
                if verbose: print('calculating dos and vvdos with lifetimes')
                wmesh, dos_tau, vvdos_tau = BTPDOS(eig_fine, vvband, erange=erange, npts=npts,
                                                      scattering_model=tau_fine, mode=dos_method)
                #store results
                tau_temp = self.tmesh[itemp]
                app(TransportResult(wmesh,dos_tau,vvdos_tau,self.fermi,tau_temp,
                                    self.volume,self.nelect,tau_temp=tau_temp,margin=margin))

        return TransportResultRobot(boltztrap_results)

    def __str__(self):
        return self.to_string()

    def to_string(self, verbose=2):
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__,mark="="))
        app("equivalent points: {}".format(self.nequivalences))
        app("real space mesh:   {}".format(self.rmesh))
        app("lpratio:           {}".format(self.lpratio))
        return "\n".join(lines)
