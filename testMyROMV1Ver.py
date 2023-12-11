from OpenFoamData import OpenFoamData
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# -- information about case
# testedCases = ['./meshInSt/mS_40p','./meshInSt/mS_50p','./meshInSt/mS_75p','./meshInSt/mS_120p']
testedCases = ['../bCyl_l_3V3']
timeSamples = np.linspace(3000,10600,3).astype(int)
timeSamples = np.linspace(3000,12316,3).astype(int)
# timeSamples = [10600]
# timeSamples = [12316]
# timeSamples = [500]
startTime = 0.9
# endTime = 1.0
endTime = 7.8
# endTime = 6.8
storage = 'PPS'
procFields = ['U']
plochaNazevLst = ["%s_plochaHor3.vtk" % procFields[0]]
plochaNazevLst = ["%s_plochaVer.vtk" % procFields[0]]
# plochaNazevLst = ["%s_plochaHor3.vtk" % procFields[0], "%s_plochaVer.vtk" % procFields[0]]
nameOfTheResultsFolder = 'flowAnalysisPy'

# -- what to do?
takePODmatricesFromFiles = True
# takePODmatricesFromFiles = False
loadFromNumpyFile = True
# loadFromNumpyFile = False
onlyXY = True
onlyXY = False
withConv = True
withConv = False
symFluc = True
# symFluc = False
indFromFile = True
# indFromFile = False
flipDirs = ['y','z']
# flipDirs = []
prepareK = True
# prepareK = False
makeSpectraChronos = True
writeModes = False
writeModes = True

# -- writing stuff 
nModes = 60
nModes = 20

# -- how many modes compare in convergence
modesToComp = 8


############################################################################################################
# -------------------------   script    --------------------------------------------------------------------
############################################################################################################

# -- folder with results
newRes = '230817_res_conv%s_symFluc_%s'%(withConv, symFluc)
nameOfTheResultsFolder = '%s_onlyXY_%s'%(nameOfTheResultsFolder, onlyXY)

# -- in all testCases
for case in testedCases:
    caseDir = case
    for plochaNazev in plochaNazevLst:
        outDir = '%s/%s_%s_time%g_%g_%s'%(caseDir,nameOfTheResultsFolder,plochaNazev.split('.')[0], startTime, endTime,storage)
        
        # -- load openFoam data
        oFData = OpenFoamData(caseDir, startTime, endTime, storage, procFields, outDir)

        # -- load data
        if not loadFromNumpyFile:
            oFData.loadYsFromOFData(plochaName = plochaNazev,onlyXY=onlyXY)
            oFData.saveYsFromNPField()
        else:
            oFData.loadYsFromNPField()
        
        # -- calculate average of the data
        oFData.calcAvgY()
        
        # -- whole snapshot matrix
        UBox = np.copy(oFData.Ys[0])
        
        # -- initialization of the error field for graph
        errorInAvg = np.zeros(len(timeSamples))
        
        # -- working for each time sample separatelly
        for i in range(len(timeSamples)):
            timeSample = timeSamples[i]
            
            # -- calculate average matrices in this time
            UBoxTu = np.copy(UBox[:,:timeSample])
            UBoxTuAvg = np.copy(np.average(UBoxTu,axis=1))
            errorInAvg[i] = np.linalg.norm(UBoxTuAvg-oFData.avgs[0])
            
            # -- write avg field into vtk:
            if onlyXY:
                prepWrite = np.append(UBoxTuAvg[:].reshape(-1,2), np.zeros((UBoxTuAvg[:].shape[0]//2,1)), axis =1)
            else:
                prepWrite = UBoxTuAvg[:].reshape(-1,3)
            oFData.writeVtkFromNumpy('avg.vtk', [prepWrite], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/'%(oFData.outDir,newRes,timeSample))
            
            # -- prepare and save k field
            if prepareK:
                # -- turbulence kinetic energy field
                k = np.zeros(UBoxTuAvg.reshape(-1,3).shape[0])

                for colInd in range(UBoxTu.shape[1]):
                    YCol = UBoxTu[:,colInd].reshape(-1,3)
                    # print(YCol)
                    YFluc = (YCol-UBoxTuAvg.reshape(-1,3))/5
                    # k += 0.5*(YFluc[:,0]**2 + YFluc[:,1]**2)
                    k += 0.5*(YFluc[:,0]**2 + YFluc[:,1]**2 + YFluc[:,2]**2)
                k = k / UBoxTu.shape[1]
                prepWrite = np.append(k.reshape(-1,1), np.zeros((k.shape[0],2)), axis =1)
                oFData.writeVtkFromNumpy('Kavg.vtk', [prepWrite], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/'%(oFData.outDir,newRes,timeSample))

            # -- write avg field into png 
            # oFData.vizNpVecInTmplVTK(prepWrite, '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), 'testAvg.png')
            
            # -- POD
            if not takePODmatricesFromFiles:
                # -- prepare directory for modes, sing values and etas writing
                if not os.path.exists('%s/%s'%(outDir,newRes)):
                    os.mkdir('%s/%s'%(outDir,newRes))

                if not os.path.exists('%s/%s/%d'%(outDir,newRes,timeSample)):
                    os.mkdir('%s/%s/%d'%(outDir,newRes,timeSample))
                
                if not os.path.exists('%s/%s/%d/topos'%(outDir,newRes,timeSample)):
                    os.mkdir('%s/%s/%d/topos'%(outDir,newRes,timeSample))

                # -- make matrix of fluctulations
                for colInd in range(UBoxTu.shape[-1]):
                    UBoxTu[:,colInd] = UBoxTu[:,colInd] - UBoxTuAvg

                # -- convolution of the fluctulations
                if withConv:
                    t = np.linspace(0,0.0005*timeSample,timeSample)
                    if not os.path.exists('%s/%s/grafyKonvoluce'%(outDir,newRes)):
                        os.mkdir('%s/%s/grafyKonvoluce'%(outDir,newRes))
                    # gausian kernel
                    # sigmaB = (3./4) ** 0.5 / shedFreq
                    sigmaB = 0.0015 
                    kernel = 1./((2*np.pi) ** 0.5 * sigmaB) * np.exp(-(t-t[-1]/2)**2/(2*sigmaB**2)) 
                    kernel = kernel / np.linalg.norm(kernel)
                    plt.plot(t,kernel)
                    plt.savefig('%s/%s/grafyKonvoluce/%d_gaussJadro.png'%(outDir,newRes,timeSample))
                    plt.close()

                    # -- just to know what the convolution +- does
                    for i in range(3):
                        plt.plot(t,UBoxTu[i*300,:], label = '%d')
                    plt.savefig('%s/%s/grafyKonvoluce/%dpredKonvoluci.png'%(outDir,newRes,timeSample))
                    plt.close()

                    for i in range(3):
                        conv = np.convolve(UBoxTu[i*300,:],kernel,mode='same')
                        plt.plot(t,conv, label = '%d')
                    plt.savefig('%s/%s/grafyKonvoluce/%dpoKonvoluci.png'%(outDir,newRes,timeSample))
                    plt.close()

                    for i in range(UBoxTu.shape[0]):
                        UBoxTu[i,:] = np.convolve(UBoxTu[i,:],kernel,mode='same')

                # -- calculation of symmetric asymetric fluctulations + POD
                if symFluc:
                    print('Calculating symmetric and antisymmetric fluctulations.')
                    if flipDirs == []:
                        Usym, UAsym = oFData.UsymUAsym(UBoxTu, plochaName = plochaNazev,onlyXY=onlyXY, indFromFile = indFromFile,flipDirs=flipDirs)
                        
                        # oFData.writeVtkFromNumpy('flucSym.vtk', [Usym[:,0].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/testFluc/'%(oFData.outDir,newRes,timeSample))
                        # oFData.writeVtkFromNumpy('flucASym.vtk', [UAsym[:,0].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/testFluc/'%(oFData.outDir,newRes,timeSample))
                        
                        # -- symmetric/antisymmetric POD and save results
                        modesSym, singValsSym, chronosSym = oFData.POD(Usym)
                        modesASym, singValsASym, chronosASym = oFData.POD(UAsym)
                        np.save('%s/%s/%d/modesSym.npy'%(outDir,newRes,timeSample),modesSym)
                        np.save('%s/%s/%d/singValsSym.npy'%(outDir,newRes,timeSample),singValsSym)
                        np.save('%s/%s/%d/chronosSym.npy'%(outDir,newRes,timeSample),chronosSym)
                        np.save('%s/%s/%d/modesASym.npy'%(outDir,newRes,timeSample),modesASym)
                        np.save('%s/%s/%d/singValsASym.npy'%(outDir,newRes,timeSample),singValsASym)
                        np.save('%s/%s/%d/chronosASym.npy'%(outDir,newRes,timeSample),chronosASym)
                        
                        # -- write singular values:
                        oFData.writeSingVals(singValsSym,'%s/%s'%(outDir,newRes), timeSample, name='singValsSym')
                        oFData.writeSingVals(singValsASym,'%s/%s'%(outDir,newRes), timeSample, name='singValsASym')
                    
                    else:
                        USymSym, USymASym, UASymSym, UASymASym = oFData.UsymUAsym(UBoxTu, plochaName = plochaNazev,onlyXY=onlyXY, indFromFile = indFromFile,flipDirs=flipDirs)
                        
                        # oFData.writeVtkFromNumpy('flucSymSym.vtk', [USymSym[:,0].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/testFluc/'%(oFData.outDir,newRes,timeSample))
                        
                        # -- symmetric/antisymmetric POD and save results
                        modesSymSym, singValsSymSym, chronosSymSym = oFData.POD(USymSym)
                        modesSymASym, singValsSymASym, chronosSymASym = oFData.POD(USymASym)
                        modesASymSym, singValsASymSym, chronosASymSym = oFData.POD(UASymSym)
                        modesASymASym, singValsASymASym, chronosASymASym = oFData.POD(UASymASym)
                        np.save('%s/%s/%d/modesSymSym.npy'%(outDir,newRes,timeSample),modesSymSym)
                        np.save('%s/%s/%d/singValsSymSym.npy'%(outDir,newRes,timeSample),singValsSymSym)
                        np.save('%s/%s/%d/chronosSymSym.npy'%(outDir,newRes,timeSample),chronosSymSym)
                        np.save('%s/%s/%d/modesSymASym.npy'%(outDir,newRes,timeSample),modesSymASym)
                        np.save('%s/%s/%d/singValsSymASym.npy'%(outDir,newRes,timeSample),singValsSymASym)
                        np.save('%s/%s/%d/chronosSymASym.npy'%(outDir,newRes,timeSample),chronosSymASym)
                        np.save('%s/%s/%d/modesASymSym.npy'%(outDir,newRes,timeSample),modesASymSym)
                        np.save('%s/%s/%d/singValsASymSym.npy'%(outDir,newRes,timeSample),singValsASymSym)
                        np.save('%s/%s/%d/chronosASymSym.npy'%(outDir,newRes,timeSample),chronosASymSym)
                        np.save('%s/%s/%d/modesASymASym.npy'%(outDir,newRes,timeSample),modesASymASym)
                        np.save('%s/%s/%d/singValsASymASym.npy'%(outDir,newRes,timeSample),singValsASymASym)
                        np.save('%s/%s/%d/chronosASymASym.npy'%(outDir,newRes,timeSample),chronosASymASym)
                        
                        
                        # -- write singular values:
                        oFData.writeSingVals(singValsSymSym,'%s/%s'%(outDir,newRes), timeSample, name='singValsSymSym')
                        oFData.writeSingVals(singValsSymASym,'%s/%s'%(outDir,newRes), timeSample, name='singValsSymASym')
                        oFData.writeSingVals(singValsASymSym,'%s/%s'%(outDir,newRes), timeSample, name='singValsASymSym')
                        oFData.writeSingVals(singValsASymASym,'%s/%s'%(outDir,newRes), timeSample, name='singValsASymASym')

                else:
                    # -- POD and save results
                    modes, singVals, chronos = oFData.POD(UBoxTu)
                    np.save('%s/%s/%d/modes.npy'%(outDir,newRes,timeSample),modes)
                    np.save('%s/%s/%d/singVals.npy'%(outDir,newRes,timeSample),singVals)
                    np.save('%s/%s/%d/chronos.npy'%(outDir,newRes,timeSample),chronos)
                    
                    # -- write singular values 
                    oFData.writeSingVals(singVals,'%s/%s'%(outDir,newRes), timeSample, name='singVals')
            
            # -- load POD stuff from file
            else:
                if symFluc:
                    if flipDirs == []:
                        modesSym = np.load('%s/%s/%d/modesSym.npy'%(outDir,newRes,timeSample))
                        singValsSym = np.load('%s/%s/%d/singValsSym.npy'%(outDir,newRes,timeSample))
                        chronosSym = np.load('%s/%s/%d/chronosSym.npy'%(outDir,newRes,timeSample))
                        modesASym = np.load('%s/%s/%d/modesASym.npy'%(outDir,newRes,timeSample))
                        singValsASym = np.load('%s/%s/%d/singValsASym.npy'%(outDir,newRes,timeSample))
                        chronosASym = np.load('%s/%s/%d/chronosASym.npy'%(outDir,newRes,timeSample))
                    else:
                        modesSymSym = np.load('%s/%s/%d/modesSymSym.npy'%(outDir,newRes,timeSample))
                        singValsSymSym = np.load('%s/%s/%d/singValsSymSym.npy'%(outDir,newRes,timeSample))
                        chronosSymSym = np.load('%s/%s/%d/chronosSymSym.npy'%(outDir,newRes,timeSample))
                        modesSymASym = np.load('%s/%s/%d/modesSymASym.npy'%(outDir,newRes,timeSample))
                        singValsSymASym = np.load('%s/%s/%d/singValsSymASym.npy'%(outDir,newRes,timeSample))
                        chronosSymASym = np.load('%s/%s/%d/chronosSymASym.npy'%(outDir,newRes,timeSample))
                        modesASymSym = np.load('%s/%s/%d/modesASymSym.npy'%(outDir,newRes,timeSample))
                        singValsASymSym = np.load('%s/%s/%d/singValsASymSym.npy'%(outDir,newRes,timeSample))
                        chronosASymSym = np.load('%s/%s/%d/chronosASymSym.npy'%(outDir,newRes,timeSample))
                        modesASymASym = np.load('%s/%s/%d/modesASymASym.npy'%(outDir,newRes,timeSample))
                        singValsASymASym = np.load('%s/%s/%d/singValsASymASym.npy'%(outDir,newRes,timeSample))
                        chronosASymASym = np.load('%s/%s/%d/chronosASymASym.npy'%(outDir,newRes,timeSample))
                        
                else:
                    modes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSample))
                    singVals = np.load('%s/%s/%d/singVals.npy'%(outDir,newRes,timeSample))
                    chronos = np.load('%s/%s/%d/chronos.npy'%(outDir,newRes,timeSample)) 
        
            # -- write modes
            if writeModes:
                for i in range(nModes):
                    if symFluc:
                        if flipDirs == []:
                            if onlyXY:
                                prepWriteSym = np.append(modesSym[:,i].reshape(-1,2), np.zeros((modesSym[:,i].shape[0]//2,1)), axis =1)
                                prepWriteASym = np.append(modesASym[:,i].reshape(-1,2), np.zeros((modesASym[:,i].shape[0]//2,1)), axis =1)
                            else:
                                prepWriteSym = modesSym[:,i].reshape(-1,3)
                                prepWriteASym = modesASym[:,i].reshape(-1,3)
                            oFData.writeVtkFromNumpy('mode%dSym.vtk'%(i+1), [prepWriteSym], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                            oFData.writeVtkFromNumpy('mode%dASym.vtk'%(i+1), [prepWriteASym], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                        else:
                            oFData.writeVtkFromNumpy('mode%dSymSym.vtk'%(i+1), [modesSymSym[:,i].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                            oFData.writeVtkFromNumpy('mode%dSymASym.vtk'%(i+1), [modesSymASym[:,i].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                            oFData.writeVtkFromNumpy('mode%dASymSym.vtk'%(i+1), [modesASymSym[:,i].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                            oFData.writeVtkFromNumpy('mode%dASymASym.vtk'%(i+1), [modesASymASym[:,i].reshape(-1,3)], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                    else:
                        if onlyXY:
                            prepWrite = np.append(modes[:,i].reshape(-1,2), np.zeros((modes[:,i].shape[0]//2,1)), axis =1)
                        else:
                            prepWrite = modes[:,i].reshape(-1,3)
                        oFData.writeVtkFromNumpy('mode%d.vtk'%(i+1), [prepWrite], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))

            # -- chronos spectra 
            if makeSpectraChronos:
                if symFluc:
                    if flipDirs == []:
                        oFData.writeChronosSpectra(chronosASym, '%s/%s'%(outDir,newRes), timeSample, 'etaASymSpct',nModes=nModes)
                        oFData.writeChronosSpectra(chronosSym, '%s/%s'%(outDir,newRes), timeSample, 'etaSymSpct',nModes=nModes)
                    else:
                        oFData.writeChronosSpectra(chronosSymSym, '%s/%s'%(outDir,newRes), timeSample, 'etaSymSymSpct',nModes=nModes)
                        oFData.writeChronosSpectra(chronosSymASym, '%s/%s'%(outDir,newRes), timeSample, 'etaSymASymSpct',nModes=nModes)
                        oFData.writeChronosSpectra(chronosASymSym, '%s/%s'%(outDir,newRes), timeSample, 'etaASymSymSpct',nModes=nModes)
                        oFData.writeChronosSpectra(chronosASymASym, '%s/%s'%(outDir,newRes), timeSample, 'etaASymASymSpct',nModes=nModes)
                    # oFData.phaseDiagrams(chronosASym, '%s/%s'%(outDir,newRes), timeSample, 'etaEtaASymSpct',nModes=nModes)
                    # oFData.phaseDiagrams(chronosSym, '%s/%s'%(outDir,newRes), timeSample, 'etaSymSpct',nModes=nModes)
            
        # -- convergence of modes and mean values
        # errModes = np.empty((0,modesToComp))
        # refModes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSamples[-1]))[:,:modesToComp]
        # for i in range(len(timeSamples)):
        #     modes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSamples[i]))[:,:modesToComp]
        #     errTu = np.empty(0)
        #     for i in range(modesToComp):
        #         chyba = np.linalg.norm(np.abs(modes[:,i])-np.abs(refModes[:,i]))
        #         errTu = np.append(errTu, chyba)
        #     err = np.append(errModes, errTu.reshape(1,-1), axis=0)
        
        # for i in range(modesToComp):
        #     plt.plot(timeSamples[:-1], err[:-1,i],label='mode %d'%(i+1))
        # plt.legend()
        # plt.savefig('%s/%s/errCompModes.png'%(outDir,newRes))  
        
        plt.close()
        plt.plot(timeSamples[:-1], errorInAvg[:-1])
        plt.legend()
        plt.savefig('%s/%s/errCompAvgsU.png'%(outDir,newRes))  


#         # oFData.writeSpectraChronos(nChronos=len(oFData.timeLst),svFig='sp_n')
#         # oFData.writeChronos(nChronos=len(oFData.timeLst),svFig='etaNN_n')

#         # oFData.vizSpectraSumChronos(nChronos=10,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=20,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=50,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=100,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=200,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=1000,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=2000,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=3000,svFig='Spectra_sum/spct')
#         # oFData.vizSpectraSumChronos(nChronos=4000,svFig='Spectra_sum/spct')
#         #
#         # oFData.vizChronos(nChronos=4,svFig='chronos')
#         # oFData.vizSpectraChronos(nChronos=4,svFig='spectraEta')

#         #----------------------
#         # calculate TKE
#         #----------------------
#         # if not onlyXY:
#         #     U_avg = np.reshape(oFData.avgs[0],(-1,3))
#         #     U_act = np.reshape(oFData.Ys[0],(-1,3,len(oFData.timeLst)))
#         #     kFromU = np.zeros((U_avg.shape[0],len(oFData.timeLst)))
#         # # for j in range(U_avg.shape[0]):
#         # #     # for i in range(len(oFData.timeLst)):
#         # #         # print(kFromU[:,i].shape,U_avg.shape,U_act[:,:,i].shape)
#         # #         # kFromU[j,i] = np.linalg.norm(U_avg[j] - U_act[j,:,i])/2
#         # #         # kFromU[j,i] = ((U_avg[j,0] - U_act[j,0,i])**2 + (U_avg[j,1] - U_act[j,1,i])**2 + (U_avg[j,2] - U_act[j,2,i])**2)/2
#         # #     kFromU[j,:] = (((U_avg[j,0] - U_act[j,0,:]))**2 + ((U_avg[j,1] - U_act[j,1,:]))**2 + ((U_avg[j,2] - U_act[j,2,:]))**2)/2
#         # # kFromU_avg = np.average(kFromU,axis=1)
#         # # kFromOF_avg = oFData.avgs[1]
#         # # kVysledek = kFromU_avg+kFromOF_avg
#         # # data = oFData.writeField(kVysledek/np.linalg.norm(kVysledek),'k','avg_k',caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)
#         # # data = oFData.writeField(kFromU_avg,'k','avg_k_fromU',caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)
        
#         #     kFromU = np.zeros((U_avg.shape[0]))
#         #     kFromUDiv = np.zeros((U_avg.shape[0]))
#         # # print(U_act.shape)
#         # # print(U_avg.shape)
#         #     for j in range(U_avg.shape[0]):
#         #     # for i in range(len(oFData.timeLst)):
#         #         # print(kFromU[:,i].shape,U_avg.shape,U_act[:,:,i].shape)
#         #         # kFromU[j,i] = np.linalg.norm(U_avg[j] - U_act[j,:,i])/2
#         #         # kFromU[j,i] = ((U_avg[j,0] - U_act[j,0,i])**2 + (U_avg[j,1] - U_act[j,1,i])**2 + (U_avg[j,2] - U_act[j,2,i])**2)/2
#         #     # print(np.average((U_avg[j,2] - U_act[j,2,:])**2))    
#         #         kFromU[j] =   (np.average((U_avg[j,0] - U_act[j,0,:])**2) + 
#         #                     np.average((U_avg[j,1] - U_act[j,1,:])**2) +
#         #                     np.average((U_avg[j,2] - U_act[j,2,:])**2))/2
#         #         kFromUDiv[j] = (np.average((U_avg[j,0] - U_act[j,0,:])**2/25) + 
#         #                     np.average((U_avg[j,1] - U_act[j,1,:])**2) +
#         #                     np.average((U_avg[j,2] - U_act[j,2,:])**2))/2
#         # # kFromU_avg = np.average(kFromU,axis=1)
#         #     kFromU_avg = kFromU
#         #     kFromOF_avg = oFData.avgs[1]
#         #     kVysledek = kFromU_avg/25+kFromOF_avg/5
#         #     data = oFData.writeField(kVysledek,'k','avg_k',caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)
#         #     data = oFData.writeField(kFromU_avg,'k','avg_k_fromU',caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)
#         #     data = oFData.writeField(kFromUDiv,'k','avg_k_fromUDivX',caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)

#         #----------------------
#         # writing fields (avgs)
#         #----------------------
#         # for i in range(len(procFields)):
#         #     oFData.writeField(oFData.avgs[i],procFields[i],'avg_%s_%d'%(procFields[i],0),caseDir,outDir='%s/avgs/'%(outDir),plochaName = plochaNazev)

#         # # #----------------------
#         # # # writing fields (topos)
#         # # #----------------------
#         # for i in range(30):
#         #     oFData.writeField(oFData.modes[0][:,i],procFields[0],'Topos%d_%s'%(i+1,procFields[0]),caseDir,outDir='%s/topos/'%(outDir),plochaName = plochaNazev)

#         #----------------------
#         # writing sum topos
#         #----------------------
#         # sums = [10,100,1000,3000,6000]
#         # # sums = [10]
#         # mode = np.copy(oFData.avgs[0])
#         # nModes = 1
#         # od = 49
#         # for suma in sums:
#         #     for j in range(od,od+nModes):
#         #         for i in range(suma):
#         #             mode += oFData.modes[0][:,i]*oFData.chronos[0][i,j]
#         #         oFData.writeField(mode,procFields[0],'Topos%d_%s_time_%d_sums_%d'%(i+1,procFields[0],j,suma),caseDir,outDir='%s/toposSum/'%(outDir),plochaName = plochaNazev)
#         #         mode = np.copy(oFData.avgs[0])



#         # #---------------------
#         # #compute and write rPOD (topos)
#         # #---------------------
#         # oFData.rPOD(singValsFile='%s/singValsRND'%(outDir))
#         # for i in range(1):
#         #     oFData.writeField(-oFData.randmodes[0][:,i],procFields[0],'Topos%d_%s_rnd'%(i+1,procFields[0]),caseDir,outDir='%s/topos/'%(outDir),plochaName = plochaNazev)
#         #     oFData.writeField((oFData.modes[0][:,i]+oFData.randmodes[0][:,i])/oFData.modes[0][:,i],procFields[0],'Topos%d_%s_reldiff'%(i+1,procFields[0]),caseDir,outDir='%s/topos/'%(outDir),plochaName = plochaNazev)

#         #-----------------------
#         #write spectra
#         #-----------------------
#         # oFData.writeSpectraChronos(nChronos=30,svFig='spectraEta')
#         # oFData.writeChronos(nChronos=30,svFig='etaNN')


# # for i in range(len(procFields)):
# #     mod1cas0 = oFData.avgs[i]
# #     data = oFData.writeField(mod1cas0,procFields[i],'avg_%s_%d'%(procFields[i],0),caseDir,outDir='%s/1000'%(caseDir))
# # for i in range(100):
# #     mod1cas0 = oFData.avgs[0] + oFData.modes[0][:,0]*oFData.chronos[0][0,i]
# #     mod2cas0 = mod1cas0 + oFData.modes[0][:,1]*oFData.chronos[0][1,i]
# #     mod3cas0 = mod2cas0 + oFData.modes[0][:,2]*oFData.chronos[0][2,i]
# #     mod4cas0 = mod3cas0 + oFData.modes[0][:,3]*oFData.chronos[0][3,i]
# #     oFData.writeField(mod1cas0,'U','akum_1',caseDir,outDir='%s/%d'%(caseDir,100+i),data=data)
# #     oFData.writeField(mod2cas0,'U','akum_2',caseDir,outDir='%s/%d'%(caseDir,100+i),data=data)
# #     oFData.writeField(mod3cas0,'U','akum_3',caseDir,outDir='%s/%d'%(caseDir,100+i),data=data)
# #     oFData.writeField(mod4cas0,'U','akum_4',caseDir,outDir='%s/%d'%(caseDir,100+i),data=data)


# # data = oFData.writeField(oFData.modes[0][:,0],'U','PsiU_%d'%0,caseDir,outDir='%s/100'%(caseDir))
# # for i in range(1,4):
# #     oFData.writeField(oFData.modes[0][:,i],'U','PsiU_%d'%i,caseDir,outDir='%s/100'%(caseDir),data=data)

# # data = oFData.writeField(oFData.modes[1][:,0],'p','PsiP_%d'%0,caseDir,outDir='%s/100'%(caseDir))
# # for i in range(4):
# #     oFData.writeField(oFData.modes[1][:,i],'p','PsiP_%d'%i,caseDir,outDir='%s/100'%(caseDir),data=data)

# #----------------------
# # run ROM
# #----------------------
# # oFData.ROM(0,4,'U',howToCreate = 'load')