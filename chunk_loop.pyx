""" Placeholder module to temporarily store a function 
that updates the histogram of pair counts. 
"""

import numpy as np

def chunked_distance_loop(x1, x2, chunk_size):
    """ Implement Manodeep Sinha's trick of avoiding 
    cache misses by looping in chunks. 
    """
    Npts1 = len(x1)
    Npts2 = len(x2)
    result = np.zeros(Npts1*Npts2)

    # Set up the chunks
    chunk1_bins = np.arange(0, Npts1, chunk_size).astype(int)
    Nchunk1 = len(chunk1_bins)
    chunk1_bins = np.append(chunk1_bins, None)

    chunk2_bins = np.arange(0, Npts2, chunk_size).astype(int)
    Nchunk2 = len(chunk2_bins)
    chunk2_bins = np.append(chunk2_bins, None)

    slice_array1 = np.zeros(Nchunk1, dtype=object)
    for ichunk in range(Nchunk1):
        slice_array1[ichunk] = slice(chunk1_bins[ichunk], chunk1_bins[ichunk+1])

    slice_array2 = np.zeros(Nchunk2, dtype=object)
    for ichunk in range(Nchunk2):
        slice_array2[ichunk] = slice(chunk2_bins[ichunk], chunk2_bins[ichunk+1])

    counter = 0
    for ichunk1 in range(Nchunk1):
        x1chunk = x1[slice_array1[ichunk1]]
        for ichunk2 in range(Nchunk2):
            x2chunk = x2[slice_array2[ichunk2]]

            Ntot_ichunk1_ichunk2 = len(x1chunk)*len(x2chunk)
            for a in x1chunk:
                for b in x2chunk:
                    result[counter] = b-a
                    counter += 1
                    
    return counter
