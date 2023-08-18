from OpenFoamData import OpenFoamData
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# testedCases = ['40','50','75']
testedCases = ['./meshInSt/mS_40p','./meshInSt/mS_50p','./meshInSt/mS_75p','./meshInSt/mS_120p'] #,'./bCyl_l_3']
# testedCases = ['./meshInSt/mS_40p','./meshInSt/mS_50p','./meshInSt/mS_75p','./meshInSt/mS_120p','./bCyl_l_3']
testedCases = ['../bCyl_l_3V3']



timeSamples = np.linspace(1000,9000,9).astype(int)
timeSamples = [500]
# timeSamples = [7000]
takePODmatricesFromFiles = False
# takePODmatricesFromFiles = True
loadFromNumpyFile = False
loadFromNumpyFile = True
modesToComp = 8
startTime = 0.9
endTime = 1.9
endTime = 6
storage = 'PPS'
procFields = ['U']
onlyXY = True
# onlyXY = False
plochaNazevLst = ["%s_plochaHor3.vtk" % procFields[0]]
# plochaNazevLst = ["%s_plochaVer.vtk" % procFields[0]]
# plochaNazevLst = ["%s_plochaHor3_filtered.vtk" % procFields[0]]
# timeSamples = [500]
# testedCases = ['./bCyl_l_3V2']
# testedCases = ['75']

# -- convolution of fluctulation
withConv = True
withConv = False
shedFreq = 69
kernLeng = int((1/69) / 0.0005)

# -- symmetric and antisymmetric fluctulations
symFluc = True

newRes = '230817_res_conv%s_symFluc_%s'%(withConv, symFluc)

for case in testedCases:
    caseDir = case
    for plochaNazev in plochaNazevLst:
        outDir = '%s/avgsPODResFrom_%s_time%g_%g_%s'%(caseDir,plochaNazev.split('.')[0], startTime, endTime,storage)
        if onlyXY:
            outDir = '%s/XYonly_avgsPODResFrom_%s_time_%g_%g_%s'%(caseDir,plochaNazev.split('.')[0], startTime, endTime,storage)
        
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

        # # -- write data
        # prepWrite = np.append(oFData.avgs[0].reshape(-1,2), np.zeros((oFData.avgs[0].shape[0]//2,1)), axis =1)
        # oFData.writeVtkFromNumpy('U_avg.vtk', [prepWrite], '%s/postProcessing/sample/3.90006723/U_plochaHor3.vtk'%oFData.caseDir, '%s/avgs/'%oFData.outDir)

        
        # -- POD
        # oFData.POD(singValsFile='%s/singVals'%(outDir))
        # print(oFData.modes.shape)
        # oFData.vizSpectraPts(svFig='Spectra_sum/spct_ptsV')
        UBox = np.copy(oFData.Ys[0])
        errorInAvg = np.zeros(len(timeSamples))
        for i in range(len(timeSamples)):
            timeSample = timeSamples[i]
            # averageMatrices in this time
            UBoxTu = np.copy(UBox[:,:timeSample])
            UBoxTuAvg = np.copy(np.average(UBoxTu,axis=1))
            errorInAvg[i] = np.linalg.norm(UBoxTuAvg-oFData.avgs[0])
            if not takePODmatricesFromFiles:
                for colInd in range(UBoxTu.shape[-1]):
                    UBoxTu[:,colInd] = UBoxTu[:,colInd] - UBoxTuAvg

                if withConv:
                    t = np.linspace(0,0.0005*timeSample,timeSample)
                    for i in range(3):
                        # print(UBoxTu.shape)
                        plt.plot(t,UBoxTu[i*300,:], label = '%d')
                    plt.savefig('grafyKonvoluce/%dpredKonvoluci.png'%timeSample)
                    plt.close()

                    # gausian kernel
                    # sigmaB = (3./4) ** 0.5 / shedFreq
                    sigmaB = 0.0015 
                    kernel = 1./((2*np.pi) ** 0.5 * sigmaB) * np.exp(-(t-t[-1]/2)**2/(2*sigmaB**2)) 
                    # kernel = kernel / np.sum(kernel)
                    kernel = kernel / np.linalg.norm(kernel)
                    # kernel = signal.windows.gaussian(t, std=(sigmaB), sym=True)
                    # print(np.sum(kernel))
                    plt.plot(t,kernel)
                    plt.savefig('grafyKonvoluce/%d_gaussJadro.png'%timeSample)
                    plt.close()

                    for i in range(3):
                        conv = np.convolve(UBoxTu[i*300,:],kernel,mode='same')
                        plt.plot(t,conv, label = '%d')
                    plt.savefig('grafyKonvoluce/%dpoKonvoluci.png'%timeSample)
                    plt.close()
                    # window = signal.windows.gaussian(len(t), std=sigmaB, sym=True)
                    # window = signal.windows.gaussian(len(t), std=0.1, sym=True)
                    # window = signal.windows.hamming((10))
                    # print(np.sum(window),np.linalg.norm(window))
                    # plt.close()
                    # plt.plot(t,window)
                    # plt.savefig('grafyKonvoluce/%d_window.png'%timeSample)
                    for i in range(UBoxTu.shape[0]):
                        UBoxTu[i,:] = np.convolve(UBoxTu[i,:],kernel,mode='same')
                    
                if symFluc:
                    print('Calculating symmetric and antisymmetric fluctulations.')
                    oFData.symmetricAntiSymmericPOD(UBoxTu)
                
                print('Running POD, processed matrix size is: (%d,%d)'%UBoxTu.shape)

                PsiBox,sBox,_ = np.linalg.svd(UBoxTu, full_matrices=False)   
                _ = None     
                etaMat   = (PsiBox.T).dot(UBoxTu)

                modes = (PsiBox)
                singVals = (sBox)
                chronos = (etaMat)

                if not os.path.exists('%s/%s'%(outDir,newRes)):
                    os.mkdir('%s/%s'%(outDir,newRes))

                if not os.path.exists('%s/%s/%d'%(outDir,newRes,timeSample)):
                    os.mkdir('%s/%s/%d'%(outDir,newRes,timeSample))
                
                if not os.path.exists('%s/%s/%d/topos'%(outDir,newRes,timeSample)):
                    os.mkdir('%s/%s/%d/topos'%(outDir,newRes,timeSample))

                with open('%s/%s/%d/modes.npy'%(outDir,newRes,timeSample),'wb') as fl:
                    np.save(fl,modes)
                with open('%s/%s/%d/singVals.npy'%(outDir,newRes,timeSample),'wb') as fl:
                    np.save(fl,singVals)
                with open('%s/%s/%d/chronos.npy'%(outDir,newRes,timeSample),'wb') as fl:
                    np.save(fl,chronos)
            else:
                modes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSample))
                singVals = np.load('%s/%s/%d/singVals.npy'%(outDir,newRes,timeSample))
                chronos = np.load('%s/%s/%d/chronos.npy'%(outDir,newRes,timeSample))

            # write results:
            with open('%s/%s/%d/singVals.dat'%(outDir,newRes,timeSample),'w') as fl:
                fl.writelines('x\tsingVal\tsingValLomSumasingVal\tsingValsqLomSumasingValsq\n')
                for j in range(len(singVals)):
                    fl.writelines('%d\t%g\t%g\t%g\n'%(j,singVals[j],singVals[j]/np.sum(singVals),singVals[j]**2/np.sum(singVals**2)))
                plt.plot(singVals**2/np.sum(singVals**2),label='%d'%timeSample)
                plt.yscale('log')
                plt.xscale('log')
                plt.ylim(10e-5,1)
                plt.legend()
                plt.savefig('%s/%s/%d/singVals.png'%(outDir,newRes,timeSample))
                # plt.close()
            
            for i in range(30):
                if onlyXY:
                    # oFData.writeField(modes[:,i],procFields[0],'Topos%d_%s'%(i+1,procFields[0]),caseDir,outDir='%s/%s/%d/topos/'%(outDir,newRes,timeSample),plochaName = plochaNazev)
                    prepWrite = np.append(modes[:,i].reshape(-1,2), np.zeros((modes[:,i].shape[0]//2,1)), axis =1)
                    # oFData.writeVtkFromNumpy('mode%d.vtk'%(i+1), [prepWrite], '%s/postProcessing/sample/3.90006723/U_plochaHor3_filtered.vtk'%oFData.caseDir, '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
                    oFData.writeVtkFromNumpy('mode%d.vtk'%(i+1), [prepWrite], '%s/postProcessing/sample/3.90006723/U_plochaHor3.vtk'%oFData.caseDir, '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))
            if onlyXY:
                prepWrite = np.append(UBoxTuAvg[:].reshape(-1,2), np.zeros((UBoxTuAvg[:].shape[0]//2,1)), axis =1)
                oFData.writeVtkFromNumpy('avg.vtk', [prepWrite], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/'%(oFData.outDir,newRes,timeSample))
            else:
                prepWrite = UBoxTuAvg[:].reshape(-1,3)
                oFData.writeVtkFromNumpy('avg.vtk', [prepWrite], '%s/postProcessing/sample/3.90006723/%s'%(oFData.caseDir,plochaNazev), '%s/%s/%d/'%(oFData.outDir,newRes,timeSample))
            # oFData.writeField(UBoxTuAvg[:],procFields[0],'avg_%s'%(procFields[0]),caseDir,outDir='%s/%s/%d/'%(outDir,newRes,timeSample),plochaName = plochaNazev)
                # prepWrite = np.append(modes[:,i].reshape(-1,2), np.zeros((modes[:,i].shape[0]//2,1)), axis =1)
                # oFData.writeVtkFromNumpy('mode%d.vtk'%(i+1), [prepWrite], '%s/postProcessing/sample/3.90006723/U_plochaHor3_filtered.vtk'%oFData.caseDir, '%s/%s/%d/toposes/'%(oFData.outDir,newRes,timeSample))

        plt.close()
        err = np.empty((0,modesToComp))
        refModes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSamples[-1]))[:,:modesToComp]
        for i in range(len(timeSamples)):
            modes = np.load('%s/%s/%d/modes.npy'%(outDir,newRes,timeSamples[i]))[:,:modesToComp]
            errTu = np.empty(0)
            for i in range(modesToComp):
                chyba = np.linalg.norm(np.abs(modes[:,i])-np.abs(refModes[:,i]))
                errTu = np.append(errTu, chyba)
            err = np.append(err, errTu.reshape(1,-1), axis=0)
        
        for i in range(modesToComp):
            plt.plot(timeSamples, err[:,i],label='mode %d'%(i+1))
        plt.legend()
        plt.savefig('%s/%s/errCompModes.png'%(outDir,newRes))  
        
        plt.close()
        plt.plot(timeSamples, errorInAvg)
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