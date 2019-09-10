import numpy as np

num_elec = 24 
electrodes = np.linspace(1, num_elec , num_elec, endpoint=True, dtype = np.int8)

seq = np.zeros((10000,4), dtype = np.int8) # preallocate sequence
meas_counter = 0

def rows_uniq_elems(seq):
    seq_sorted = np.sort(seq, axis = -1)
    return seq[(seq_sorted[...,1:] != seq_sorted[...,:-1]).all(-1)]

#---   ---   ---   ---   ---   ---   --- dipole-dipole
def add_skip(skip, meas_counter, seq, electrodes):
    print('adding skip ', skip, 'from measurement number ', meas_counter + 1)
    for a in electrodes:
        b = a + 1 + skip
        for m in electrodes:
            n = m + 1 + skip
            seq[meas_counter, :] = [a, b, m, n]
            meas_counter += 1
    return(seq, meas_counter)


seq, meas_counter = add_skip(1, meas_counter, seq, electrodes)
seq, meas_counter = add_skip(2, meas_counter, seq, electrodes)
seq, meas_counter = add_skip(4, meas_counter, seq, electrodes)

#---   ---   ---   ---   ---   ---   ---  cross-well
print('adding cross-well from measurement number ', meas_counter + 1)
for a in electrodes[0:12]:
    b = a + num_elec / 2 # set b on the well opposite to a
    for m in electrodes[0:12]:
        n = m + num_elec / 2 # set n in the well opposite to m
        seq[meas_counter, :] = [a, b, m, n]
        meas_counter += 1

#---   ---   ---   ---   ---   ---   ---  cleaning
# force restart counting for num > num_elec
bigger = np.where(seq > num_elec)
seq[bigger] = seq[bigger] - num_elec

# remove rows that use twice the same electrode
seq_clean = rows_uniq_elems(seq)

#---   ---   ---   ---   ---   ---   ---  write
# just the sequence
np.savetxt('sequence.txt', seq_clean, fmt = '%i %i %i %i')

# schd sequence
len_seq_clean = seq_clean.shape[0]
id_meas = np.linspace(1, len_seq_clean, len_seq_clean, endpoint = True, dtype = np.int16)
ones = np.ones_like(id_meas)
seq_schd = np.column_stack((id_meas, ones, seq_clean[:,0], ones, seq_clean[:, 1], ones, seq_clean[:, 2], ones, seq_clean[:, 3]))
print(seq_schd)
np.savetxt('sequence_schd.txt', seq_schd, fmt = '%i %i %i %i %i %i %i %i %i')