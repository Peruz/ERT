import numpy as np
import sys 
import os
import re
from subprocess import call
import matplotlib.pyplot as plt

try:
    # get argument input for sequence and lab data 
    NameLab = [l for l in sys.argv if '.data' in l][0]
    # get most recent (lastest created, no highest number) fop-model.vtk
    allVTKs = sorted([s for s in os.listdir() if s.startswith('fop-model')])
    NameVTK = max(allVTKs, key = lambda x: os.stat(x).st_mtime)
except IndexError:
    sys.exit('missing input argument with extension ".data"')
except ValueError:
    sys.exit('no "fop-model.vtk" in the directory, make sure to run BERT with "SAVEALOT=1"')

# mesh and synthetic data names are expected to be standard, according to BERT: 
NameBMS = 'meshSec.bms'
NameSyn = 'dcmodOut.ohm'

# create directory for comparison, move necessary files and set as cd
mkdir = 'rm -rf LabSynComparison && mkdir LabSynComparison'
call(mkdir, shell=True)

mvData = 'cp -p ' + NameLab + ' ./LabSynComparison/' + NameLab
print(mvData)
call(mvData, shell=True)

mvVTK = 'cp -p ' + NameVTK + ' ./LabSynComparison/' + NameVTK
print(mvVTK)
call(mvVTK, shell=True)

mvBMS = 'cp -p ' + NameBMS + ' ./LabSynComparison/' + NameBMS
print(mvBMS)
call(mvBMS, shell=True)

os.chdir('./LabSynComparison')

def readLab(NameLab):
    print('reading')
    coordinates = []
    data = []
    read_coord = 0
    read_data = 0
    with open (NameLab) as fid:
        for line in fid:
            if line[0] == '#':
                if 'x y z' in line:
                    print('reading coordinates')
                    read_coord = 1 # start coordinates
                if 'a b m n' in line:
                    print('reading data')
                    read_coord = 0 # stop coordinates
                    read_data = 1 # start data
            else:
                if read_coord == 1:
                    line_coord = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',line)
                    coordinates.append(line_coord)
                if read_data == 1:
                    line_data = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',line)
                    data.append(line_data)
    print(data)
    del coordinates[-1] # del number of measurements
    del data[-1] # del zero at the end
    LabCoord = np.array(coordinates, dtype = 'f' )
    LabData = np.array(data, dtype = 'f') 
    return(LabCoord, LabData)

def writeSeq(LabData):
    sequence = LabData[:,0:4]
    coordinates = LabCoord
    num_elec = np.array(len(coordinates))
    num_meas = np.array(len(sequence))
    precoordinates_str =np.array(['# x y z'])
    presequence_str = np.array(['# a b m n'])
    with open ('seq.shm','ab') as f_handle:
        np.savetxt (f_handle,num_elec[None], fmt = '%i')
        np.savetxt (f_handle,precoordinates_str, fmt = '%s')
        np.savetxt (f_handle,coordinates, fmt = '%3.3f')
        np.savetxt (f_handle,num_meas[None], fmt = '%i')
        np.savetxt (f_handle,presequence_str, fmt = '%s')
        np.savetxt (f_handle,sequence, fmt = '%i %i %i %i')

# read ".data" and write sequence for forward modeling
LabCoord, LabData = readLab(NameLab)
writeSeq(LabData)

def readVTK(NameVTK):
    rhoVect = []
    fid = open(NameVTK)
    tables = {}
    for line in fid:
        if 'F-op-model' in line:
            for i, i_cont in enumerate(fid):
                tables.update({i:i_cont})
    rhoStr = tables[1]
    rhoList = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', rhoStr)
    rhoNP = np.array(rhoList, dtype = 'f')
    rhoLen = len(rhoNP)
    print('number of resistivity values: ', rhoLen)
    print('revistivity vector: ', rhoNP)
    number_column = np.linspace(1, rhoLen, rhoLen)
    rhoMap = np.column_stack((number_column, rhoNP))
    with open ('rho.map','ab') as f_handle:
        np.savetxt(f_handle,rhoMap, fmt = '%i %f')
    return (rhoMap)

# read "fop-model.vtk" and write "rho.map" for forward simulation
rhoMap = readVTK(NameVTK)

# run forward modeling
DCMOD ='dcmod -S -a rho.map -s seq.shm -o dcmodOut meshSec.bms' 
print(DCMOD,'\nthis takes few seconds')
call(DCMOD, shell=True)

def readSyn(NameSyn):
    coordinates = []
    data = []
    read_coord = 0
    read_data = 0
    fid = open (NameSyn)
    for line in fid:
        if line[0] == '#':
            if line[2] == 'x':
                read_coord = 1 # start coordinates
            if line[2] == 'a':
                read_coord = 0 # stop coordinates
                read_data = 1 # start data
        else:
            if read_coord == 1:
                line_coord = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',line)
                coordinates.append(line_coord)
            if read_data == 1:
                line_data = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',line)
                data.append(line_data)
    del coordinates[-1]
    del data[-1]
    SynCoord = np.array(coordinates, dtype = 'f' )
    SynData = np.array(data, dtype = 'f') 
    return(SynCoord,SynData)

# read synthetic data output from forward modeling
SynCoord, SynData = readSyn(NameSyn)

# small check between lab and syn data, through coordinates
if (LabCoord != SynCoord).any():
    print('LAb and Syn coordinates are different')

lenLD, lenSD = len(LabData), len(SynData)

# other check between lab and syn data
if lenLD != lenSD:
    raise Exception('Different number of data, cannot continue with plotting')

# plot comparison
rLD, rSD = LabData[:,4], SynData[:,4]
deltaD = rLD - rSD
maxLD = max(rLD)
maxSD = max(rSD)
maxD = max(maxLD, maxSD)
minD = min(min(rLD), min(rSD))
numD = np.linspace(1,lenLD,lenLD)

font = {'family':'serif','style':'normal','weight':'normal', 'size':12}
plt.rc('font', **font)

plt.figure(1, figsize = (12,8))
plt.subplot(121)
plt.plot(rLD, rSD, 'og')
plt.xlabel('LabData [ohm]', fontsize = 14)
plt.ylabel('SynData [ohm]', fontsize = 14)
plt.xlim(minD-2, maxD+2)
plt.ylim(minD-2, maxD+2)
ax= plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.subplot(122)
plt.plot(numD,rLD,'or', label = 'Lab R [ohm]')
plt.plot(numD,rSD,'ob', label = 'Syn R [ohm]')
plt.xlabel('Data number', fontsize = 14)
plt.ylabel('Resistance [ohm]', fontsize = 14)
plt.legend()
plt.grid(which = 'major', axis = 'both', linewidth = 2)
plt.tight_layout()
plt.savefig('LabSynComparison.png')
plt.show()
