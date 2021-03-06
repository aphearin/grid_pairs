#!/usr/bin/env python
# cython: profile=True

from __future__ import print_function, division
import numpy as np
from numba import vectorize, double
cimport cython
cimport numpy as np
from libc.math cimport fabs, fmin
import sys

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def npairs(data1, data2, rbins, Lbox):
    """
    Calculate the number of pairs with separations less than or equal to rbins[i].
    
    Parameters
    ----------
    data1: array_like
        3 by N array of spatial positions. 
            
    data2: array_like
        3 by N array of spatial positions. 
            
    rbins : array_like
        Array of boundaries defining the bins in which pairs are counted. 
        len(rbins) = Nrbins + 1.
            
    Lbox: float 
        Size of the periodic box. 
            
    Returns
    -------
    N_pairs : array of length len(rbins)
        number counts of pairs
    """
    
    #c definitions
    cdef int nbins = len(rbins)
    cdef np.ndarray[np.float64_t, ndim=1] crbins = np.ascontiguousarray(rbins,dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] counts = np.zeros(nbins,dtype=np.int)
    
    #build grids for data1 and data2
    grid1 = cube_grid(data1[0,:], data1[1,:], data1[2,:], Lbox, np.max(rbins))
    grid2 = cube_grid(data2[0,:], data2[1,:], data2[2,:], Lbox, np.max(rbins))

    #square radial bins
    crbins = crbins**2.0
        
    #more c definitions used inside loop
    cdef int i,j,icell1,icell2, k
    cdef double dx, dy, dz, d
    cdef np.ndarray[np.float64_t, ndim=1] x_icell1, y_icell1, z_icell1
    cdef np.ndarray[np.float64_t, ndim=1] x_icell2, y_icell2, z_icell2
    

    #Loop over all subvolumes in grid1
    for icell1 in range(grid1.num_divs**3):
        #calculate progress
        progress = icell1/(grid1.num_divs**3)*100
        print("    {0:.2f} %%".format(progress),end='\r')
        sys.stdout.flush()
        
        #extract the points in the cell
        x_icell1, y_icell1, z_icell1 = (grid1.x[grid1.slice_array[icell1]],\
                                        grid1.y[grid1.slice_array[icell1]],\
                                        grid1.z[grid1.slice_array[icell1]])
        
        #get the list of neighboring cells
        ix1, iy1, iz1 = np.unravel_index(icell1,(grid1.num_divs,\
                                              grid1.num_divs,\
                                              grid1.num_divs))
        adj_cell_arr = grid1.adjacent_cells(ix1, iy1, iz1)

        # Compute the cells that will need to be shifted to respect PBCs
        ix2_move, x2_move = shift_subvolume(ix1, grid1.num_divs, Lbox)
        iy2_move, y2_move = shift_subvolume(iy1, grid1.num_divs, Lbox)
        iz2_move, z2_move = shift_subvolume(iz1, grid1.num_divs, Lbox)

        
        #Loop over each of the 27 subvolumes neighboring, including the current cell.
        for icell2 in adj_cell_arr:

            ix2, iy2, iz2 = np.unravel_index(icell2,(grid2.num_divs,\
                                                  grid2.num_divs,\
                                                  grid2.num_divs))

            #extract the points in the cell
            x_icell2 = grid2.x[grid2.slice_array[icell2]]
            y_icell2 = grid2.y[grid2.slice_array[icell2]]
            z_icell2 = grid2.z[grid2.slice_array[icell2]]

            # Shift the points, as necessary
            if ix2==ix2_move:
                x_icell2 += x2_move
            if iy2==iy2_move:
                y_icell2 += y2_move
            if iz2==iz2_move:
                z_icell2 += z2_move

            #loop over points in grid1's cell
            for i in range(0,len(x_icell1)):
                #loop over points in grid2's cell
                for j in range(0,len(x_icell2)):

                    #calculate the square distance dsq
                    dx = x_icell1[i] - x_icell2[j]
                    dy = y_icell1[i] - y_icell2[j]
                    dz = z_icell1[i] - z_icell2[j]
                    dsq = dx*dx+dy*dy+dz*dz

                    ### Calculate counts in bins
                    k = nbins-1
                    while dsq<=crbins[k]:
                        counts[k] += 1
                        k=k-1
                        if k<0: break

    return counts


def shift_subvolume(idim1, num_divs, Lbox):
    """ For a 1d index idim1 of a subvolume, compute the 
    1d indices of subvolumes that (may) need to be shifted 
    in order to respect PBC distances. 

    Parameters 
    ----------
    idim1 : int 
        1d index of a subvolume

    num_divs : int 
        Number of divisions each dimension of the box 
        has been divided into

    Lbox : float 
        Size of the box. Defines the periodic boundary condition. 

    Returns 
    -------
    idim2_move : int 
        1d index of the subvolume that needs to be shifted

    dim2_move : float 
        Distance by which the subvolume needs to be shifted, 
        including sign. 

    """

    if idim1==0:
        idim2_move = num_divs-1
        dim2_move = -Lbox
    elif idim1==num_divs-1:
        idim2_move = 0
        dim2_move = Lbox
    else:
        idim2_move = -1
        dim2_move = 0

    return idim2_move, dim2_move


class cube_grid():

    def __init__(self, x, y, z, Lbox, cell_size):
        """
        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 
        
        Lbox : float
            Length scale defining the Lboxic boundary conditions

        cell_size : float 
            The approximate cell size into which the box will be divided. 
        """

        self.cell_size = cell_size
        self.Lbox = Lbox
        self.num_divs = np.floor(Lbox/float(cell_size)).astype(int)
        self.dL = Lbox/float(self.num_divs)
        
        #build grid tree
        idx_sorted, slice_array = self.compute_cell_structure(x, y, z)
        self.x = np.ascontiguousarray(x[idx_sorted],dtype=np.float64)
        self.y = np.ascontiguousarray(y[idx_sorted],dtype=np.float64)
        self.z = np.ascontiguousarray(z[idx_sorted],dtype=np.float64)
        self.slice_array = slice_array

    def compute_cell_structure(self, x, y, z):
        """ 
        Method divides the periodic box into regular, cubical subvolumes, and assigns a 
        subvolume index to each point.  The returned arrays can be used to efficiently 
        access only those points in a given subvolume. 

        Parameters 
        ----------
        x, y, z : arrays
            Length-Npts arrays containing the spatial position of the Npts points. 

        Returns 
        -------
        idx_sorted : array
            Array of indices that sort the points according to the dictionary 
            order of the 3d subvolumes. 

        slice_array : array 
            array of slice objects used to access the elements of x, y, and z 
            of points residing in a given subvolume. 

        Notes 
        -----
        The dictionary ordering of 3d cells where :math:`dL = L_{\\rm box} / 2` 
        is defined as follows:

            * (0, 0, 0) <--> 0

            * (0, 0, 1) <--> 1

            * (0, 1, 0) <--> 2

            * (0, 1, 1) <--> 3

        And so forth. Each of the Npts thus has a unique triplet, 
        or equivalently, unique integer specifying the subvolume containing the point. 
        The unique integer is called the *cellID*. 
        In order to access the *x* positions of the points lying in subvolume *i*, 
        x[idx_sort][slice_array[i]]. 

        In practice, because fancy indexing with `idx_sort` is not instantaneous, 
        it will be more efficient to use `idx_sort` once to sort the x, y, and z arrays 
        in-place, and then access the sorted arrays with the relevant slice_array element. 
        This is the strategy used in the `retrieve_tree` method. 

        """

        ix = np.floor(x/self.dL).astype(int)
        iy = np.floor(y/self.dL).astype(int)
        iz = np.floor(z/self.dL).astype(int)

        particle_indices = np.ravel_multi_index((ix, iy, iz),\
                                               (self.num_divs,\
                                                self.num_divs,\
                                                self.num_divs))
        
        idx_sorted = np.argsort(particle_indices)
        bin_indices = np.searchsorted(particle_indices[idx_sorted], 
                                      np.arange(self.num_divs**3))
        bin_indices = np.append(bin_indices, None)
        
        slice_array = np.empty(self.num_divs**3, dtype=object)
        for icell in range(self.num_divs**3):
            slice_array[icell] = slice(bin_indices[icell], bin_indices[icell+1], 1)
            
        return idx_sorted, slice_array
    
    def adjacent_cells(self, *args):
        """ 
        Given a subvolume specified by the input arguments,  
        return the length-27 array of cellIDs of the neighboring cells. 
        The input subvolume can be specified either by its ix, iy, iz triplet, 
        or by its cellID. 

        Parameters 
        ----------
        ix, iy, iz : int, optional
            Integers specifying the ix, iy, and iz triplet of the subvolume. 
            If ix, iy, and iz are not passed, then ic must be passed. 

        ic : int, optional
            Integer specifying the cellID of the input subvolume
            If ic is not passed, the ix, iy, and iz must be passed. 

        Returns 
        -------
        result : int array
            Length-27 array of cellIDs of neighboring subvolumes. 

        Notes 
        -----
        If one argument is passed to `adjacent_cells`, this argument will be 
        interpreted as the cellID of the input subvolume. 
        If three arguments are passed, these will be interpreted as 
        the ix, iy, iz triplet of the input subvolume. 

        """

        ixgen, iygen, izgen = np.unravel_index(np.arange(3**3), (3, 3, 3)) 

        if len(args) >= 3:
            ix, iy, iz = args[0], args[1], args[2]
        elif len(args) == 1:
            ic = args[0]
            ix, iy, iz = np.unravel_index(ic, (self.num_divs, self.num_divs, self.num_divs))

        ixgen = (ixgen + ix - 1) % self.num_divs
        iygen = (iygen + iy - 1) % self.num_divs
        izgen = (izgen + iz - 1) % self.num_divs

        return np.ravel_multi_index((ixgen, iygen, izgen), 
            (self.num_divs, self.num_divs, self.num_divs))

