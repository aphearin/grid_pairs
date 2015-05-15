from numba import autojit

@autojit
def autojit_update_histogram(x, bins, hist):
    k = len(bins)-1
    while x <= bins[k]:
        hist[k] += 1
        k=k-1
        if k<0: break
    return hist