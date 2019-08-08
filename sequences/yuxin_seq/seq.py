import numpy as np
tot_elec_num = 32
skips = [2, 4]
sing_borehole = 16
seq = np.zeros((5000, 4), dtype = int)
elec = np.linspace(1, tot_elec_num, tot_elec_num, dtype = int)
quadripole_num = 0
#dipole-dipole skip 2
for skip in skips:
    for a in elec:
        b = a + skip + 1
        for m in elec:
            n = m + skip + 1
            quadripole = [a,b,m,n]
            if len(set(quadripole)) < 4:
                continue 
            seq[quadripole_num, :] = quadripole
            quadripole_num += 1

#opposite
b1 = np.linspace(1, sing_borehole, sing_borehole, dtype = int)
for a in b1:
    b = a + 16
    for m in b1:
        if m == a:
            continue
        n = m + 16
        quadripole = [a,b,m,n]
        if len(set(quadripole)) < 4:
            continue 
        seq[quadripole_num, :] = quadripole
        quadripole_num += 1


#clean up sequence
mask_elec_bigger32 = np.where(seq > 32)
seq[mask_elec_bigger32] = seq[mask_elec_bigger32] - 32
mask_zeros = (seq == 0).all(1)
seq = seq[~mask_zeros]

# add columns to seq

id_num = np.linspace(1, len(seq), len(seq))
ones = np.ones(len(seq))

seq_out = np.column_stack((id_num, ones, seq[:,0], ones, seq[:,1], ones, seq[:,2], ones, seq[:,3]))

# header
# inches to m
convert = 0.0254

x1 = np.zeros(16)
x2 = x1 + 24*convert
x = np.concatenate((x1, x2))
print(x)
y = np.zeros(32)
print(y)
first = 5*convert
last = first + 15*3.5*convert
z1 = np.linspace(first, last, 16)
z = np.concatenate((z1,z1))
print(z)

#
elec_num = np.linspace(1,32,32)
print(elec_num)
cable = np.ones(32)
print(cable)
header = np.column_stack((cable, elec_num, x, y, z))
print(header)
np.savetxt('dd_sequence.txt', seq_out, fmt = '%i %i %i %i %i %i %i %i %i')
np.savetxt('header.txt', header, fmt = '%i %i %2.4f %2.4f %2.4f' )

