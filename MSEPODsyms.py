from OpenFoamData import OpenFoamData
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import signal
from scipy import sparse

# -- information about case
# testedCases = ['./meshInSt/mS_40p','./meshInSt/mS_50p','./meshInSt/mS_75p','./meshInSt/mS_120p']
testedCases = ['../bCyl_l_3V3']
# timeSamples = np.linspace(3000,10600,3).astype(int)
# timeSamples = np.linspace(3000,12316,3).astype(int)
# timeSamples = [10600]
# timeSamples = [2000,4000,8000,12000]
timeSamples = [2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000]
# timeSamples = [2000]
# timeSamples = [100]
# timeSamples = [500]
startTime = 0.9
endTime = 1.0
endTime = 7.8
# endTime = 6.8
storage = 'PPS'
procFields = ['U']
plochaNazevLst = ["%s_plochaHor3.vtk" % procFields[0]]
# plochaNazevLst = ["%s_plochaVer.vtk" % procFields[0]]
# plochaNazevLst = ["%s_plochaHor3.vtk" % procFields[0], "%s_plochaVer.vtk" % procFields[0]]
# nameOfTheResultsFolder = 'flowAnalysisPyIntrpltd'
# nameOfTheResultsFolderLst = ['flowAnalysisPyIntrpltd','flowAnalysisPyIntrpltd2']
nameOfTheResultsFolderLst = ['flowAnalysisPyIntrpltd2']
nameOfTheResultsFolderLst = ['flowAnalysisPyIntrpltd3']
# nameOfTheResultsFolderLst = ['flowAnalysisPyIntrpltd2']
# nameOfTheResultsFolderLst = ['flowAnalysisPyIntrpltd3']
# nameOfTheResultsFolder = 'flowAnalysisPyIntrpltd2'
# nameOfTheResultsFolder = 'flowAnalysisPyIntrpltd3'

# -- interpolated geom info
# targetVTK = 'intTest.vtk'                                               # name of the interpolated vtk geometry    
# startPoint = [0.007525, -0.0304145, 0.125]
# size = [0.105235-0.007525+0.000842, 2*0.0304145, 0]
# nCells = [59,38,1]
# nCells = [59*2,38*2,1]
# nCells = [59*4,38*4,1]

# -- what to do?
takePODmatricesFromFiles = True
# takePODmatricesFromFiles = False
loadFromNumpyFile = True
# loadFromNumpyFile = False
onlyXY = True
# onlyXY = False
withConv = True
withConv = False
symFluc=True
# indFromFile = True
indFromFile = False
# flipDirs = ['y','z']
flipDirs = []
prepareK = True
prepareK = False
makeSpectraChronos = True
makeSpectraChronos = False
writeModes = False
# writeModes = True
# mergeSingVals = True
mergeSingVals = False
createIntpltdGeom = True
createIntpltdGeom = False
createMatrixForInt = True
createMatrixForInt = False
interpolateFld = True


# -- writing stuff 
nModes = 60
nModes = 20

# -- how many modes compare in convergence
modesToComp = 8

# -- np with error
# mOfInt = [1,2,3,4,5,6,7,8] 
mOfIntSym = [1,5] 
mOfIntASym = [1,8] 
MSE = np.zeros((len(mOfIntSym)+len(mOfIntASym),len(timeSamples),len(nameOfTheResultsFolderLst)))

############################################################################################################
# -------------------------   script    --------------------------------------------------------------------
############################################################################################################

# -- folder with results
newRes = '230817_res_conv%s_symFluc_%s'%(withConv, symFluc)


# -- in all testCases
for case in testedCases:
    caseDir = case
    for plochaNazev in plochaNazevLst:
        for nameOfTheResultsFolderInd in range(len(nameOfTheResultsFolderLst)):
            
            nameOfTheResultsFolder = '%s_onlyXY_%s'%(nameOfTheResultsFolderLst[nameOfTheResultsFolderInd], onlyXY)
            outDir = '%s/%s_%s_time%g_%g_%s'%(caseDir,nameOfTheResultsFolder,plochaNazev.split('.')[0], startTime, endTime,storage)
            
            # -- reference modes
            modesSymRef = np.load('%s/%s/%d/modesSym.npy'%(outDir,newRes,timeSamples[-1]))
            # singValsSymRef = np.load('%s/%s/%d/singValsSym.npy'%(outDir,newRes,timeSamples[-1]))
            # chronosSymRef = np.load('%s/%s/%d/chronosSym.npy'%(outDir,newRes,timeSamples[-1])) 
            modesASymRef = np.load('%s/%s/%d/modesASym.npy'%(outDir,newRes,timeSamples[-1]))
            # singValsSymRef = np.load('%s/%s/%d/singValsASym.npy'%(outDir,newRes,timeSamples[-1]))
            # chronosSymRef = np.load('%s/%s/%d/chronosASym.npy'%(outDir,newRes,timeSamples[-1])) 
            
            # -- working for each time sample separatelly
            for i in range(len(timeSamples)):
                timeSample = timeSamples[i]
                
                modesSym = np.load('%s/%s/%d/modesSym.npy'%(outDir,newRes,timeSample))
                # singValsSym = np.load('%s/%s/%d/singValsSym.npy'%(outDir,newRes,timeSample))
                # chronosSym = np.load('%s/%s/%d/chronosSym.npy'%(outDir,newRes,timeSample)) 
                
                modesASym = np.load('%s/%s/%d/modesASym.npy'%(outDir,newRes,timeSample))
                # singValsASym = np.load('%s/%s/%d/singValsASym.npy'%(outDir,newRes,timeSample))
                # chronosASym = np.load('%s/%s/%d/chronosASym.npy'%(outDir,newRes,timeSample))
                
                for j in range(len(mOfIntSym)):
                    mOfIntTu = mOfIntSym[j] - 1
                    
                    modeTu = np.abs(modesSym[:,mOfIntTu])
                    modeRef = np.abs(modesSymRef[:,mOfIntTu])
                    
                    MSE[j, i, nameOfTheResultsFolderInd] = np.sum((modeTu-modeRef)**2) 
                
                for j in range(len(mOfIntASym)):
                    mOfIntTu = mOfIntASym[j] - 1
                    
                    modeTu = np.abs(modesASym[:,mOfIntTu])
                    modeRef = np.abs(modesASymRef[:,mOfIntTu])
                    
                    MSE[j+len(mOfIntSym), i, nameOfTheResultsFolderInd] = np.sum((modeTu-modeRef)**2) 

matplotlib_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

with open('%s/%s/MSE3Syms.dat'%(outDir,newRes), 'w') as fl:
    fl.writelines('nSn\t')
    for i in mOfIntSym:
        fl.writelines('colSym%d\t'%i)
    for i in mOfIntASym:
        fl.writelines('colASym%d\t'%i)
    fl.writelines('\n')
    # for i in range(MSE.shape[0]):
    #     plt.plot(timeSamples[:-1],MSE[i,:-1,0],color=matplotlib_colors[i],label="%d 1"%mOfInt[i])
    for i in range(len(timeSamples)-1):
        fl.writelines('%g\t'%timeSamples[i])
        for j in range((MSE.shape[0])):
            fl.writelines('%g\t'%MSE[j,i,0])
        fl.writelines('\n')
# for i in range(MSE.shape[0]):
#     plt.plot(timeSamples[:-1],MSE[i,:-1,1],'--',color=matplotlib_colors[i],label="%d 2"%mOfInt[i])
    
plt.legend()
plt.yscale('log')
print('%s/%s/MSE1.png'%(outDir,newRes))
plt.savefig('%s/%s/MSE1.png'%(outDir,newRes))
                