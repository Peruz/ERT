import numpy as np
sequence=[]
electrodes=range(1,65)
electrodes=np.reshape(electrodes,(8,8))
print(electrodes)
# GRADIENT
# vertical and horizontal
for i in range (8):
    a = electrodes[i,0]
    b = electrodes[i,-1]
    for j in range (1,6):
        m = electrodes[i,j]
        n = electrodes[i,j+1]
        quadripole = [a,b,m,n]
        sequence.append(quadripole)   
for i in range (8):
    a = electrodes[0,i]
    b = electrodes[-1,i]
    for j in range (1,6):
        m = electrodes [j,i]
        n = electrodes [j+1,i]
        quadripole = [a,b,m,n]
        sequence.append(quadripole)

# diagonals
for i in range (-4,5):
    diagonal = np.diag(electrodes,k=i)
    print(diagonal)
    diagonal_NumElec = len(diagonal)
    a = diagonal[0]
    b = diagonal[-1]
    for j in range (1,diagonal_NumElec-2):
        quadripole = [a,b,diagonal[j],diagonal[j+1]]
        sequence.append(quadripole)
        
electrodes_flip = np.fliplr(electrodes) # flip electrodes matrix to get other diagonals
print(electrodes_flip)

for i in range (-4,5):
    diagonal = np.diag(electrodes_flip,k=i)
    print(diagonal)
    diagonal_NumElec = len(diagonal)
    a = diagonal[0]
    b = diagonal[-1]
    print(diagonal)
    for j in range (1,diagonal_NumElec-2):
        quadripole = [a,b,diagonal[j],diagonal[j+1]]
        sequence.append(quadripole)

""" WENNER """

#Hoz
for i in range(8):
    line = electrodes[i,:]
    print("hor",i,line)
    for j in range(5):
        quadripole =[line[j],line[j+3],line[j+1],line[j+2]]
        sequence.append(quadripole)
    
    

#Ver
for i in range(8):
    line = electrodes[:,i]
    print("vert",i,line)
    for j in range(5):
        quadripole =[line[j],line[j+3],line[j+1],line[j+2]]
        sequence.append(quadripole)

#diagonals
for i in range(-3,4): # skipping shortest diagonals (already in Grad) 
    diagonal = np.diag(electrodes,k=i)
    print("W_diag",i,diagonal)
    diagonal_NumElec = len(diagonal)
    for j in range(diagonal_NumElec-3):
        quadripole = [diagonal[j],diagonal[j+3],diagonal[j+1],diagonal[j+2]]
        sequence.append(quadripole)
for i in range(-3,4):
    diagonal = np.diag(electrodes_flip,k=i)
    print("W_diag_flip",i,diagonal)
    diagonal_NumElec = len(diagonal)
    for j in range(diagonal_NumElec-3):
        quadripole = [diagonal[j],diagonal[j+3],diagonal[j+1],diagonal[j+2]]
        sequence.append(quadripole)  

SeqDir = np.array(sequence)
Ad, Bd, Md, Nd = SeqDir[:,0], SeqDir[:,1], SeqDir[:,2], SeqDir[:,3]
SeqRec = np.column_stack((Md, Nd, Ad, Bd))
Seq = np.vstack((SeqDir,SeqRec))
#A,B,M,N = sequence[:,0],sequence[:,1],sequence[:,2],sequence[:,3]
#sequenceRec = [M,N,A,B]
print(len(Seq))
np.savetxt("Seq.txt", Seq, fmt = '%i %i %i %i')
