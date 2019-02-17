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
import scipy as sp
import pandas as pd
import abipy.core.abinit_units as abu
from itertools import product
from collections import OrderedDict

from monty.string import marquee
from monty.termcolor import cprint
from abipy.tools import duck
from abipy.core.structure import Structure
from abipy.tools.plotting import add_fig_kwargs, get_ax_fig_plt, get_axarray_fig_plt #, set_axlims, set_visible, set_ax_xylabels
from abipy.tools.decorators import timeit

__all__ = [
    "TransportResult",
    "TransportResultRobot",
]

class TransportResult():
    """
    Container for Transport results
    This class can be initialized from a Boltztrap2 calculation or from Abinit
    Provides a object oriented interface to BoltztraP2
    for plotting, storing and analysing the results
    """
    _attrs = ['_N','_mobility','_L0','_L1','_L2','_sigma','_seebeck','_kappa']
    _properties = ['N','mobility','sigma','seebeck','kappa','powerfactor','L0','L1','L2']

    def __init__(self,wmesh,dos,vvdos,fermi,el_temp,volume,nelect,tau_temp=None,nsppol=1,margin=0.1):

        self.fermi   = fermi
        self.volume  = volume
        self.wmesh   = np.array(wmesh)
        idx_margin   = int(margin*len(wmesh))
        self.mumesh  = self.wmesh[idx_margin:-(idx_margin+1)]
        self.el_temp = el_temp
        self.nsppol  = nsppol
        self.nelect  = nelect

        #Temperature fix
        if self.el_temp < 1:
            cprint("Boltztrap does not handle 0K well.\n"
                   "I avoid potential problems by setting all T<1K to T=1K",color="yellow")
            self.el_temp = 1

        self.tau_temp = tau_temp

        self.dos = dos
        self.vvdos = vvdos

    @classmethod
    def from_evk(cls,filename,el_temp=300):
        """
        Initialize the class from an EVK.nc file containing the dos and vvdos computed by Abinit

        Args:
            el_temp: electronic temperature (in K) to use in the integrations with the Fermi-Dirac occupations
        """
        from abipy.electrons.ddk import EvkReader
        reader = EvkReader(filename)
        wmesh, dos, idos = reader.read_dos()
        wmesh, vvdos = reader.read_vvdos()

        ebands = reader.read_ebands()
        fermi = ebands.fermie*abu.eV_Ha
        nsppol = ebands.nsppol
        volume = ebands.structure.volume*abu.Ang_Bohr**3
        nelect = ebands.nelect

        #todo spin
        dos = dos[0]
        vvdos = vvdos[:,:,0]
        return cls(wmesh,dos,vvdos,fermi,el_temp,volume,nelect,tau_temp=None,nsppol=1,margin=0.1)

    @property
    def minw(self):
        return np.min(self.wmesh)

    @property
    def maxw(self):
        return np.max(self.wmesh)

    @property
    def spin_degen(self):
        return {1:2,2:1}[self.nsppol]

    @property
    def has_tau(self):
        return self.tau_temp is not None

    @property
    def N(self):
        if not hasattr(self,'_N'):
            self.compute_fermiintegrals()
        return self._N

    @property
    def L0(self):
        if not hasattr(self,'_L0'):
            self.compute_fermiintegrals()
        return self._L0

    @property
    def L1(self):
        if not hasattr(self,'_L1'):
            self.compute_fermiintegrals()
        return self._L1

    @property
    def L2(self):
        if not hasattr(self,'_L2'):
            self.compute_fermiintegrals()
        return self._L2

    @property
    def mobility(self):
        if not hasattr(self,'_mobility'):
            self.compute_onsager_coefficients()
        return self._mobility

    @property
    def sigma(self):
        if not hasattr(self,'_sigma'):
            self.compute_onsager_coefficients()
        return self._sigma

    @property
    def seebeck(self):
        if not hasattr(self,'_seebeck'):
            self.compute_onsager_coefficients()
        return self._seebeck

    @property
    def powerfactor(self):
        return self.sigma * self.seebeck**2

    @property
    def kappa(self):
        if not hasattr(self,'_kappa'):
            self.compute_onsager_coefficients()
        return self._kappa

    def del_attrs(self):
        """ Remove all the atributes so they are recomputed when requested """
        for attr in self._attrs:
            if hasattr(self,attr):
                delattr(self,attr)

    def set_el_temp(self,el_temp):
        """
        Set the electronic temperature in K

        Args:
            el_temp: electronic temperature (in K) to use in the integrations with the Fermi-Dirac occupations
        """
        self.el_temp = el_temp
        self.del_attrs()

    def set_mumesh(self,mumesh):
        """
        Set a list of mu at which to compute the transport properties

        Args:
            mumesh: an array with the list of fermi energies (in eV) at which the transport quantities should be computed
        """
        mumesh = np.array(mumesh) * abu.eV_Ha + self.fermi
        if not duck.is_listlike(mumesh): raise ValueError('The input mumesh must be a list')
        min_mumesh = np.min(mumesh)
        max_mumesh = np.max(mumesh)
        if min_mumesh < self.minw: raise ValueError('The minimum of the input mu mesh is lower than the energies mesh in DOS')
        if max_mumesh > self.maxw: raise ValueError('The maximum of the input mu mesh is higher than the energies mesh in DOS')
        self.mumesh = mumesh
        self.del_attrs()

    def compute_fermiintegrals(self):
        """Compute and store the results of the Fermi integrals"""
        import BoltzTraP2.bandlib as BL
        results = BL.fermiintegrals(self.wmesh, self.dos, self.vvdos, mur=self.mumesh, Tr=np.array([self.el_temp]))
        N, self._L0, self._L1, self._L2, self._Lm11 = results
        # Compute carrier density (N/Bohr^3)
        f = sp.interpolate.interp1d(self.mumesh,N[0])
        n0 = f(self.fermi)
        self._N = (N[0] - n0) / self.volume / ( abu.Bohr_Ang*1e-10 )**3
        #self._N = np.abs(self._N)

    def compute_onsager_coefficients(self):
        """Compute Onsager coefficients"""
        import BoltzTraP2.bandlib as BL
        L0,L1,L2 = self.L0,self.L1,self.L2
        results = BL.calc_Onsager_coefficients(L0, L1, L2, mur=self.mumesh, Tr=np.array([self.el_temp]), vuc=self.volume)
        self._sigma, self._seebeck, self._kappa, self._hall = results
        # compute carier density (N/m^3) (N is the number of electrons)
        n = self.N[None,:,None,None]
        # charge density (q/m^3)
        en = (abu.e_Cb*n)
        # compute mobility in cm^2 / (Vs)
        self._mobility = np.divide(self._sigma, en, where=en!=0) * 100**2

    def get_value_at_mu(self,what,mu,component='xx'):
        """
        Get the value of a property at a certain chemical potential (eV)

        Args
            what: quantity to plot
            mu: value of the chemical potential (eV)
            component: the component of the tensor
        """
        x = self.mumesh-self.fermi
        y = self.get_component(what,component)
        f = sp.interpolate.interp1d(x,y)
        return f(mu*abu.eV_Ha)

    def get_mu_at_n(self,n):
        """
        Get the value of the chemical potential at a given carrier concentrarion

        Args
            n: value of the carrier density (N/cm^3) positive for electrons, negative for holes
        """
        x = self.N/( abu.Bohr_Ang*1e-8 )**3
        y = (self.mumesh-self.fermi)*abu.Ha_eV
        f = sp.interpolate.interp1d(x,y)
        return f(n)

    def get_value_at_n(self,what,n,component='xx'):
        """
        Get the value of a property at a certain carrier concentration

        Args
            what: quantity to plot
            n: value of the carrier density (N/cm^3) positive for electrons, negative for holes
        """
        efermi = self.get_fermi_at_n(n)
        return self.get_value_at_mu(what,efermi,component=component)

    @staticmethod
    def from_pickle(filename):
        """Load TransportResult from a pickle file"""
        with open(filename,'rb') as f:
            instance = pickle.load(f)
        return instance

    def pickle(self,filename):
        """ Write a file with the results from the calculation """
        with open(filename,'wb') as f:
            pickle.dump(self,f)

    def istensor(self,what):
        """ Check if a certain quantity is a tensor """
        if not hasattr(self,what): return None
        return len(getattr(self,what).shape) > 2

    def deepcopy(self):
        """ Return a copy of this class """
        import copy
        return copy.deepcopy(self)

    def get_dataframe(self,components=('xx',),index=None):
        """
        Get a pandas dataframe with the results

        Args:
            mumesh: Set a certain mumesh before returning the dataframe
            tmesh:
        """
        records = []
        for component in components:
            for imu,mu in enumerate(self.mumesh):
                od = OrderedDict()
                od['mu'] = (mu-self.fermi)*abu.Ha_eV
                od['component'] = component
                for what in self._properties:
                    ylist = self.get_component(what,component)
                    od[what] = ylist[imu]
                records.append(od)
        return pd.DataFrame.from_records(records, index=index)

    def get_dataframe_fermi(self,index=None):
        """ Get dataframe for a single mu that corresponds to the Fermi energy
        """
        btr = self.deepcopy()
        btr.set_mumesh([self.fermi])
        return btr.get_dataframe(index=index)

    def get_component(self,what,component):
        i,j = abu.s2itup(component)
        return getattr(self,what)[0,:,i,j]

    def plot_dos_ax(self,ax,fontsize=8,show_fermi=False,**kwargs):
        """
        Plot the density of states on axis ax.

        Args:
            ax: |matplotlib-Axes|.
            kwargs: Passed to ax.plot
            fermi: Choose
        """
        wmesh = (self.wmesh-self.fermi) * abu.Ha_eV
        label = kwargs.pop('label',self.get_letter('dos'))
        ax.plot(wmesh,self.dos,label=label,**kwargs)
        ax.set_xlabel('Energy (eV)',fontsize=fontsize)
        if show_fermi: ax.axvline(self.fermi)

    def plot_vvdos_ax(self,ax,components=('xx',),fontsize=8,show_fermi=False,**kwargs):
        """
        Plot components of vvdos on the axis ax.

        Args:
            ax: |matplotlib-Axes|.
            components: Choose the components of the tensor to plot ['xx','xy','xz','yy',(...)]
            kwargs: Passed to ax.plot
        """
        wmesh = (self.wmesh-self.fermi) * abu.Ha_eV

        for component in components:
            i,j = abu.s2itup(component)
            label = kwargs.pop('label',"%s $_{%s}$" % (self.get_letter('vvdos'),component))
            if self.tau_temp: label += r" $\tau_T$ = %dK" % self.tau_temp
            ax.plot(wmesh,self.vvdos[i,j,:],label=label,**kwargs)
        ax.set_xlabel('Energy (eV)',fontsize=fontsize)
        if show_fermi: ax.axvline(self.fermi)

    def plot_ax(self, ax, what, components=('xx',), fontsize=8, **kwargs):
        """
        Plot a quantity for all the dopings on the axis ax.

        Args:
            ax: |matplotlib-Axes|.
            what: choose the quantity to plot can be: ['sigma','kappa','powerfactor']
            components: Choose the components of the tensor to plot ['xx','xy','xz','yy',(...)]
            colormap: Colormap used to plot the results
            kwargs: Passed to ax.plot
        """
        from matplotlib import pyplot as plt
        colormap = kwargs.pop('colormap','plasma')
        cmap = plt.get_cmap(colormap)
        color = None
        if what == 'dos':
            self.plot_dos_ax(ax,**kwargs)
            return
        if what == 'vvdos':
            self.plot_vvdos_ax(ax,components=components,**kwargs)
            return

        mumesh = (self.mumesh-self.fermi) * abu.Ha_eV

        if self.istensor(what):
            color = kwargs.pop('c',None)
            for component in components:
                y = self.get_component(what,component)
                label = kwargs.pop('label',"%s $_{%s}$ $b_T$ = %dK" % (self.get_letter(what),component,self.el_temp))
                if self.has_tau: label += r" $\tau_T$ = %dK" % self.tau_temp
                ax.plot(mumesh,y,label=label,c=color,**kwargs)
        else:
            ax.plot(mumesh,getattr(self,what), label=what, **kwargs)

        ax.set_ylabel(self.get_ylabel(what), fontsize=fontsize)
        ax.set_xlabel('Energy (eV)', fontsize=fontsize)

    def get_ylabel(self,what):
        """
        Get a label with units for the quantities stores in this object.
        """
        if self.has_tau: tau = 's^{-1}'
        else: tau = ''
        if what == 'sigma':       return r'$\sigma$ [$Sm^{-1}%s$]'%tau
        if what == 'seebeck':     return r'$S$ [$VSm^{-1}%s$]'%tau
        if what == 'kappa':       return r'$\kappa_e$ [$VJSm^{-1}%s$]'%tau
        if what == 'powerfactor': return r'$S^2\sigma$ [$VJSm^{-1}%s$]'%tau
        if what == 'mobility':    return r'$\mu_e$ [$cm^2V^{-1}%s$]'%tau
        return ''

    def get_letter(self,what):
        letters = {'sigma':      r'$\sigma$',
                   'seebeck':    r'$S$',
                   'kappa':      r'$\kappa_e$',
                   'powerfactor':r'$S^2\sigma$',
                   'vvdos':      r'$v\otimes v$',
                   'dos':        r'$n(\epsilon)$',
                   'mobility':   r'$\mu_e(\epsilon)$',
                   'L0':         r'$\mathcal{l}^{(0)}$',
                   'L1':         r'$\mathcal{l}^{(1)}$',
                   'L2':         r'$\mathcal{l}^{(2)}$'}
        return letters[what]

    @add_fig_kwargs
    def plot(self, what, colormap='plasma', components=('xx',), ax=None, fontsize=8, **kwargs):
        """
        Plot the qantity for all the temperatures as a function of the doping
        """
        ax, fig, plt = get_ax_fig_plt(ax=ax)
        self.plot_ax(ax, what, colormap=colormap, components=components, **kwargs)
        ax.legend(loc="best", shadow=True, fontsize=fontsize)
        return fig

    def to_string(self, title=None, mark="=", verbose=0):
        """
        String representation of the class
        """
        lines = []; app = lines.append
        if title is None: app(marquee(self.__class__.__name__,mark=mark))
        app("fermi:    %.5lf eV"%(self.fermi*abu.Ha_eV))
        app("mumesh:   %.5lf <-> %.5lf eV"%(self.mumesh[0]*abu.Ha_eV,self.mumesh[-1]*abu.Ha_eV))
        app("el_temp:  %.1lf K"%self.el_temp)
        if self.tau_temp: app("tau_temp: %.1lf K"%self.tau_temp)
        return "\n".join(lines)

    def __str__(self):
        return self.to_string()

class TransportResultRobot():
    """
    Robot to analyse multiple Boltztrap calculations
    Behaves as a list of TransportResult
    Provides methods to plot multiple results on a single figure
    """
    def __init__(self,results,erange=None):
        if not all([isinstance(r,TransportResult) for r in results]):
            raise ValueError('Must provide BolztrapResult instances.')

        res0 = results[0]
        #consistency check in chemical potential meshes
        #if np.any([res0.wmesh != res.wmesh for res in results]):
        #    cprint("Comparing TransportResults with different energy meshes.", color="yellow")

        #store the results
        self.results = results
        self.erange = erange

        #if not all([np.allclose(results[0].mumesh,result.mumesh) for result in results[1:]]):
        #    raise ValueError('The doping meshes of the results differ, cannot continue')
        #self.mumesh = results[0].mumesh

    def __getitem__(self,index):
        """Access the results stored in the class as a list"""
        return self.results[index]

    @property
    def tau_list(self):
        """Get all the results with tau included"""
        return [ res.tau_temp for res in self.results if res.tau_temp is not None ]

    @property
    def notau_results(self):
        """Get all the results without the tau included"""
        results = [ res for res in self.results if res.tau_temp is None ]
        if len(results) == 0: return []
        instance = self.__class__(results)
        if self.erange: instance.erange = self.erange
        return instance

    @property
    def tau_results(self):
        """Return all the results that have temperature dependence"""
        results = [ res for res in self.results if res.tau_temp ]
        if len(results) == 0: return []
        instance = self.__class__(results)
        if self.erange: instance.erange = self.erange
        return instance

    @property
    def nresults(self):
        return len(self.results)

    @staticmethod
    def from_pickle(filename):
        """
        Load results from file
        """
        with open(filename,'rb') as f:
            instance = pickle.load(f)
        return instance

    def pickle(self,filename):
        """
        Write a file with the results from the calculation
        """
        with open(filename,'wb') as f:
            pickle.dump(self,f)

    def plot_vvdos_ax(self,ax,legend=True,components=('xx',),fontsize=8,erange=None,**kwargs):
        """
        Plot the vvdos for all the results in the robot
        """
        from matplotlib import pyplot as plt
        colormap = kwargs.pop('colormap','plasma')
        cmap = plt.get_cmap(colormap)

        #set erange
        erange = erange or self.erange
        if erange is not None: ax.set_xlim(erange)

        for result in self.results:
            result.plot_vvdos_ax(ax,fontsize=fontsize,components=components,**kwargs)
        if legend: ax.legend(loc="best", shadow=True, fontsize=fontsize)

    def plot_dos_ax(self, ax1, legend=True, fontsize=8, erange=None, **kwargs):
        """
        Plot the dos for all the results in the robot
        """
        #set erange
        erange = erange or self.erange
        if erange is not None: ax1.set_xlim(erange)

        for result in self.results:
            result.plot_dos_ax(ax1,fontsize=fontsize,**kwargs)
        if legend: ax1.legend(loc="best", shadow=True, fontsize=fontsize)

    def plot_ax(self,ax1,what,components=('xx',),fontsize=8,erange=None,**kwargs):
        """
        Plot the same quantity for all the results on axis ax1

        Args:
            ax1: |matplotlib-Axes|.
            what: choose the quantity to plot can be: ['sigma','kappa','powerfactor']
            components: Choose the components of the tensor to plot ['xx','xy','xz','yy',(...)]
            erange: choose energy range of the plot
            kwargs: Passed to ax.plot
       """
        from matplotlib import pyplot as plt
        colormap = kwargs.pop('colormap','plasma')
        cmap = plt.get_cmap(colormap)

        #set erange
        erange = erange or self.erange
        if erange is not None: ax1.set_xlim(erange)

        for result in self.results:
            result.plot_ax(ax1,what,components,fontsize=fontsize,**kwargs)

    @add_fig_kwargs
    def plot_transport(self, itau_list=None, components=('xx',),
                       erange=None, ax_array=None, fontsize=8, legend=True, **kwargs):
        """
        Plot the different quantities relevant for transport for all the results in the robot
        """
        ax_array, fig, plt = get_axarray_fig_plt(ax_array,nrows=2,ncols=2)
        self.plot_ax(ax_array[0,0],'sigma',      fontsize=fontsize,**kwargs)
        self.plot_ax(ax_array[0,1],'seebeck',    fontsize=fontsize,**kwargs)
        self.plot_ax(ax_array[1,0],'kappa',      fontsize=fontsize,**kwargs)
        self.plot_ax(ax_array[1,1],'powerfactor',fontsize=fontsize,**kwargs)

        if legend:
            for ax in ax_array.flatten(): ax.legend(loc="best", shadow=True, fontsize=fontsize)

        #fig.tight_layout()
        return fig

    @add_fig_kwargs
    def plot(self,what,components=('xx',),erange=None,fontsize=8,legend=True,**kwargs):
        """
        Plot all the boltztrap results in the Robot

        Args:
            what: choose the quantity to plot can be: ['sigma','kappa','powerfactor']
            components: Choose the components of the tensor to plot ['xx','xy','xz','yy',(...)]
            erange: choose energy range of the plot
            kwargs: Passed to ax.plot
        """
        ax1, fig, plt = get_ax_fig_plt(ax=None)
        self.plot_ax(ax1,what,components=components,
                     fontsize=fontsize,erange=erange,**kwargs)
        if legend: ax1.legend(loc="best", shadow=True, fontsize=fontsize)
        return fig

    @add_fig_kwargs
    def plot_dos_vvdos(self,dos_color=None,erange=None,ax_array=None,components=('xx',),fontsize=8,legend=True,**kwargs):
        """
        Plot dos and vvdos on the same figure
        """
        ax_array, fig, plt = get_axarray_fig_plt(ax_array,nrows=2)
        self.plot_dos_ax(ax_array[0],erange=erange,legend=legend,fontsize=fontsize,**kwargs)
        self.plot_vvdos_ax(ax_array[1],components=components,erange=erange,fontsize=fontsize,legend=legend)
        return fig

    @add_fig_kwargs
    def plot_dos(self,ax=None,erange=None,fontsize=8,legend=True,**kwargs):
        """
        Plot dos for the results in the Robot
        """
        ax1, fig, plt = get_ax_fig_plt(ax=ax)
        self.plot_dos_ax(ax1,erange=erange,legend=legend,fontsize=fontsize,**kwargs)
        return fig

    def set_erange(self,emin,emax):
        """ Get an energy range based on an energy margin above and bellow the fermi level"""
        self.erange = (emin,emax)

    def unset_erange(self):
        """ Unset the energy range"""
        self.erange = None

    def get_dataframe(self,index=None):
        """
        Get a pandas dataframe from all the results in this Robot
        """
        df_list = []; app = df_list.append
        for result in self.results:
            app(result.get_dataframe(index=index))
        return pd.concat(df_list)

    def get_dataframe_fermi(self,index=None):
        """
        Get a pandas dataframe from all the results in this Robot at the Fermi energy
        """
        df_list = []; app = df_list.append
        for result in self.results:
            app(result.get_dataframe_fermi(index=index))
        return pd.concat(df_list)

    def to_string(self, verbose=0):
        """
        Return a string representation of the data in this class
        """
        lines = []; app = lines.append
        app(marquee(self.__class__.__name__,mark="="))
        app('nresults: %d'%self.nresults)
        for result in self.results:
            app(result.to_string(mark='-'))
        return "\n".join(lines)

    def set_nmesh(self,nmesh,mode="boltztrap:refine"):
        """
        Set the range in which to plot the change of the doping
        for all the results

        Args:
            mumesh: an array with the list of fermi energies (in eV) at which the transport quantities should be computed
        """
        for result in self.results:
            result.set_nmesh(nmesh,mode=mode)

    def set_mumesh(self,mumesh):
        """
        Set the range in which to plot the change of the doping
        for all the results

        Args:
            mumesh: an array with the list of fermi energies (in eV) at which the transport quantities should be computed
        """
        for result in self.results:
            result.set_mumesh(mumesh)

    def set_tmesh(self,tmesh):
        """
        Set the temperature mesh of all the results

        Args:
            tmesh: array with temperatures at which to compute the Fermi integrals
        """
        for result in self.results:
            result.set_tmesh(tmesh)

    def __str__(self):
        return self.to_string()
