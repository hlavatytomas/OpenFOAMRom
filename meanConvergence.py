# -- using OF_caseClass for the description see OF_caseClass.py
# -- imports 

from OpenFoamData import OpenFoamData

from OF_caseClass import OpenFOAMCase
import sys
import numpy as np
import os
import matplotlib.pyplot as plt

caseOutFolder = '../bCyl_l_3V3'
numpyOutDir = '../ZZ_testNumpy'
# fieldsOfInt = ['p','U']
# fieldsOfInt = ['p']
parallel = False
loadFromNumpyFile = False
onlyXY = False
fieldsOfInt = ['U']
startTime = 0.9
endTime = 6.0
plochaNazev="U_plochaHor3.vtk"

storage = 'PPS'

oFData = OpenFoamData(caseOutFolder, startTime, endTime, storage, fieldsOfInt, numpyOutDir)
# oFData.loadTimeLst()

# -- load data
if not loadFromNumpyFile:
    oFData.loadYsFromOFData(plochaName = plochaNazev,onlyXY=onlyXY)
    oFData.saveYsFromNPField()
else:
    oFData.loadYsFromNPField()

# avgDef = np.average(oFData.Ys[0],axis=1)
# avgDefs = [[2000,3000],[3000,4000],[4000,5000],[5000,6000]]
# for j in range(len(avgDefs)):
# avgDef = np.average(oFData.Ys[0][:,1000:],axis=1)
# avgRange = [500,1000,2000, 3000, 4000, 5000]
# for j in range(len(avgRange)):
    # rangeT = avgRange[j]
    # avgDef = np.average(oFData.Ys[0][:,1500:1000+rangeT],axis=1)
avgDef = np.average(oFData.Ys[0][:,0:9000],axis=1)
# avgDef = np.average(oFData.Ys[0][:,avgDefs[j][0]:avgDefs[j][1]],axis=1)

# averages = [1000,2000,3000,4000,5000,6000]
# averages = np.linspace(3500,6000,40).astype(int)
averages = np.linspace(2000,9000,9).astype(int)
normErr = np.zeros(len(averages))
for i in range(len(averages)):
    # avg = np.average(oFData.Ys[0][:,3000:averages[i]],axis=1)
    # avg = np.average(oFData.Ys[0][:,1000:averages[i]],axis=1)
    avg = np.average(oFData.Ys[0][:,0:averages[i]],axis=1)
    print(i,np.linalg.norm(avg-avgDef))
    normErr[i] = np.linalg.norm(avg-avgDef)

    # U_avg = np.reshape(avg,(-1,3))
    # U_act = np.reshape(oFData.Ys[0][:,averages[i]-1000:averages[i]],(-1,3,1000))
    # print(U_avg.shape,U_act.shape)
    # kFromU = np.zeros((U_avg.shape[0],1000))
    # for j in range(U_avg.shape[0]):
    #     kFromU[j,:] = (((U_avg[j,0] - U_act[j,0,:]))**2 + ((U_avg[j,1] - U_act[j,1,:]))**2 + ((U_avg[j,2] - U_act[j,2,:]))**2)/2
    # kFromU_avg = np.average(kFromU,axis=1)
    # print(kFromU_avg.shape)

# plt.plot(averages[:-1],normErr)

    # U write
    oFData.writeField(
        avg[:],
        fieldsOfInt[0],
        # 'avg_%d_%d'%(1000,averages[i]),
        'avg_%d_%d'%(0,averages[i]),
        caseOutFolder,
        outDir='%s/avgs_Hor_VTK_test/' % (numpyOutDir),
        plochaName = plochaNazev
    )
    # k write
    # oFData.writeField(
    #     kFromU_avg[:],
    #     'k',
    #     # 'avg_%d_%d'%(1000,averages[i]),
    #     'k_avg_%d_%d'%(averages[i]-1000,averages[i]),
    #     caseOutFolder,
    #     outDir='%s/avgs_Hor_VTK_test/' % (numpyOutDir),
    #     plochaName = plochaNazev
    # )

# plt.plot(averages[:-1],normErr,label='%d'%rangeT)
# plt.legend()
# plt.ylim((0,70))
# plt.savefig('%s/convergenceOfTheVelocityAverageV3.png'%numpyOutDir)

#print(oFData.avgs[0].shape)

#testCase = OpenFOAMCase()
#testCase.loadOFCaseFromBaseCase('/opt/openfoam10/tutorials/incompressible/icoFoam/cavity/cavity')
#testCase.changeOFCaseDir(caseOutFolder)
#testCase.copyBaseCase()
#testCase.setParameters(
#    [
#        ['system/controlDict', 'endTime', str(3), ''],
#        ['system/fvSchemes', 'default', 'Gauss SFCD', 'divSchemes'],
#    ]
#)
#testCase.replace(
#    [
#        ['constant/physicalProperties', ['0.01'], ['0.02']]
#    ]
#)
#testCase.addToDictionary(
#    [
#        ['system/fvSchemes','div(phi,U) Gauss upwind phi;\n', 'divSchemes']
#    ]
#)
#testCase.runCommands(
#    [
#        'blockMesh > log.blockMesh',
#        'icoFoam > log.icoFoam',
#    ]
#)

#oFData = OpenFoamData(caseOutFolder, 0.1, 0.2,'case', fieldsOfInt, (numpyOutDir))
# oFData = OpenFoamData(caseOutFolder, 0.15, 0.2,'VTK-parallel', fieldsOfInt, (numpyOutDir),parallel=parallel)
# oFData.loadTimeLst()

# # oFData.loadYsFromOFData()
# # oFData.saveYsFromNPField()

# # oFData.loadYsFromNPField()

# oFData.calcAvgY(avgName='avgsTest')
# oFData.reconstructVTKs(destName='avgsTest/avg.vtk')

# oFData.POD(singValsFile='%s/singVals'%(numpyOutDir))

# for i in range(30):
#     oFData.writeField(oFData.modes[0][:,i],fieldsOfInt[0],'Topos%d_%s'%(i+1,fieldsOfInt[0]),caseOutFolder,outDir='%s/1000/' % (caseOutFolder))

# for i in range(len(fieldsOfInt)):
#     oFData.writeField(
#         oFData.avgs[i],
#         fieldsOfInt[i],
#         'avg_%s'%(fieldsOfInt[i]),
#         caseOutFolder,
#         outDir='%s/1000/' % (caseOutFolder)
#     )
