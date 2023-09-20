# -- Python class to store, postProcess and analyze OpenFOAM data

# -- imports
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from math import floor
import vtk
from multiprocessing import Pool
from scipy import sparse
# import pyrennModV3 as prn

# -- custom function
from myAddFcs import *

# class to store openfoam data in python and perform ROM stuff (POD, ROM)
class OpenFoamData:
    def __init__(self, caseDir, startTime, endTime, storage, procFields, outDir,parallel=False):
        self.caseDir = caseDir                                         # where the case folder is stored
        self.startTime = startTime                                     # start time of the interest
        self.endTime = endTime                                         # end time of the interest
        self.storage = storage                                         # 1) case - case, 2) postProcessing/sample - PPS, 3) 3DVTK - VTK
        self.procFields = procFields                                   # proceded fields (python list e.g. ['U','p'])
        self.outDir = outDir                                           # where to save results
        self.Ys = []                                                   # python list with openfoam field data stored as numpy arrays
        self.mkDels = []                                               # python list with deleted dimensions in self.Ys numpy fields
        self.lenVecsNCells = []                                        # python list with fields dimensions: [[lenVec, nCells]]
        self.avgs = []                                                 # python list with openfoam averaged fields
        self.modes = []                                                # -- NOTETH: legacy python list with calculated modes
        self.chronos = []                                              # -- NOTETH: legacy python list with saved chronoses
        self.singVals = []                                             # -- NOTETH: legacy python list with singular values of the PO decomposition
        self.randmodes = []                                            # -- NOTETH: legacy python list with calculated randomized modes
        self.randchronos = []                                          # -- NOTETH: legacy python list with saved randomized chronoses
        self.randsingVals = []                                         # -- NOTETH: legacy python list with randomized singular values of the PO decomposition
        self.parallel = parallel
        print('I am creating %s OpenFOAM fields in Python in %s, in time interval from %g to %g, looking into %s.'%(str(self.procFields), self.caseDir, self.startTime, self.endTime, self.storage)) 
        # --- create out directory
        if not os.path.exists('%s/'%(self.outDir)):             
            os.makedirs('%s/'%(self.outDir))
    
    # -- function to load time and vtk list
    # -- NOTETH: only for PPS case at the moment
    def loadVTKandTimeLst(self, plochaName = "plochaHor.vtk"):
        fld = self.caseDir + '/postProcessing/sample/'
        times = sorted([time for time in os.listdir(fld) if isFloat(time)])
        times = [time for time in times if float(time) >= self.startTime and float(time) <= self.endTime]
        vtkFiles = []
        for time in times:
            if os.path.isfile('%s/%s/%s_new.vtk'%(fld, time, plochaName.split('.')[0])):
                vtkFiles.append('%s/%s/%s_new.vtk'%(fld, time, plochaName.split('.')[0]))
            elif os.path.isfile('%s/%s/%s'%(fld, time, plochaName)):
                vtkFiles.append('%s/%s/%s'%(fld, time, plochaName))
        self.timeLst = times
        self.vtkFiles = vtkFiles

    # --- function to load openfoam data for given fields into numpy arrays
    def loadYsFromOFData(self, plochaName = "plochaHor.vtk", mkDel = [], onlyXY=False):
        if self.storage == 'case':
            for i in range(len(self.procFields)):
                field = self.procFields[i]
                self.loadFieldFilesCase(field)
        elif self.storage == 'PPS':
            # -- load all times
            self.loadVTKandTimeLst(plochaName=plochaName)
            print('Loaded %d vtkFiles from time %s to time %s' % (len(self.vtkFiles), str(self.timeLst[0]), str(self.timeLst[-1])))

            # -- reference field
            ref = self.loadNumpyFromVtk(self.vtkFiles[0])[0]
            
            # -- prepare Y_npy
            with open(self.vtkFiles[0], 'r') as file:
                data = file.readlines()

            # -- get the output dimensions
            self.lenVecsNCells.append(self.getVecLengthPPS(data))
            
            # -- get matrix of snapshots
            if onlyXY:
                Y = np.empty((ref.shape[0]*2,0))
            else:
                Y = np.empty((ref.shape[0]*ref.shape[1],0))
            for vtkFile in self.vtkFiles:
                print('Working on %s' %vtkFile)
                fldLst = self.loadNumpyFromVtk(vtkFile)
                for fld in fldLst:
                    if onlyXY:
                        fld = fld[:,:2]
                    Y = np.append(Y, np.reshape(fld, (-1,1)), axis=1)
            self.Ys.append(Y)
            if onlyXY:
                self.mkDels.append(2)
    
    # --- function to save fields into numpy fields
    def saveYsFromNPField(self):
        for i in range(len(self.Ys)):
            np.save('%s/Y_%s.npy'%(self.outDir,self.procFields[i]),self.Ys[i])
            print('I have saved %s field into %s/Y_%s.npy.'%(self.procFields[i],self.outDir,self.procFields[i]))
        with open('%s/mkDels.data'%self.outDir, 'wb') as f:
            pickle.dump(self.mkDels, f)
        print('I have saved deleted vals into %s/mkDels.npy.'%(self.outDir))
        with open('%s/lenVecNCells.data'%self.outDir, 'wb') as f:
            pickle.dump(self.lenVecsNCells, f)
        print('I have saved nCells and lenVec vals into %s/mkDels.npy.'%(self.outDir))
    
    # --- function to load fields from numpy fields
    def loadYsFromNPField(self):
        for i in range(len(self.procFields)):
            self.Ys.append(np.load('%s/Y_%s.npy'%(self.outDir,self.procFields[i])))
            print('I have loaded %s field from %s/Y_%s.npy.'%(self.procFields[i],self.outDir,self.procFields[i]))
        with open('%s/mkDels.data'%self.outDir, 'rb') as f:
            self.mkDels = pickle.load(f)
        print('I have loaded deleted vals as %s.'%str(self.mkDels))
        with open('%s/lenVecNCells.data'%self.outDir, 'rb') as f:
            self.lenVecsNCells = pickle.load(f)
        print('I have loaded nCells and lenVec vals as %s.'%str(self.lenVecsNCells))

    # -- function to calculate weighting matrix for flipping the field
    def loadIndicesForFlipping(self, flip_axis = [1, -1, 1],plochaName = "plochaHor.vtk"):
        # -- load source field
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(self.vtkFiles[0])
        reader.Update()
        source = reader.GetOutput()

        # Create a filter for flipping the data
        data = reader.GetOutput()
        flip_filter = vtk.vtkTransformFilter()
        flip_transform = vtk.vtkTransform()
        flip_transform.Scale(flip_axis[0], flip_axis[1], flip_axis[2])
        flip_filter.SetTransform(flip_transform)
        flip_filter.SetInputData(data)
        flip_filter.Update()
        target = flip_filter.GetOutput()
        
        # -- locator in the target mesh 
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(target)
        locator.BuildLocator()
        
        # -- cells in target mesh 
        cellsOld = np.array(target.GetPolys().GetData()).reshape(target.GetNumberOfCells(),-1)[:,1:]
        
        # -- find indexes that transform into new one after flipping
        nOfCells = source.GetNumberOfCells()
        Wf = np.zeros((nOfCells,nOfCells))
        closestPoint = np.empty((3,3))
        for i in range(nOfCells):
            # -- for each cell in source find where it is in new vtk
            cell = source.GetCell(i)
            # if i % int(nOfCells/10) == 0:
            #     print('Working on cell %d/%d'%(i,nOfCells))
            if isinstance(cell, vtk.vtkTriangle):
                cellPoints = cell.GetPoints()
                for j in range(cell.GetNumberOfPoints()):
                    closestPoint[j,:] = cellPoints.GetPoint(j)
                clPointLst = np.array([locator.FindClosestPoint(closestPoint[i,:]) for i in range(3)])
                cellsInTarget = np.where(cellsOld == np.array([clPointLst[0]]))[0]
                for j in range(2):
                    cellIndTu = np.where(cellsOld == np.array([clPointLst[j+1]]))[0]
                    cellsInTarget = np.intersect1d(cellIndTu, cellsInTarget)
                try:
                    closestPointId = cellsInTarget[0]
                    if cellsInTarget.size != 1:
                        print('Cell %d not 1 cells in target %s'%(i,str(cellsInTarget)))
                    Wf[i,closestPointId] = 1
                except:
                    # -- for cells that have points in flipped mesh
                    # -- I will find all cells that have points of the original mesh near the found points of the flipped mesh and interpolate from these cells 
                    print('Cell %d not found' % i)
                    # if np.size(cellIndTu) != 0:
                        # print('Found inds of point%s'%str(clPointLst))
                    allCells = np.empty((0)).astype(int)
                    for j in range(clPointLst.shape[0]):
                        # allCells = np.append(allCells,np.where(cellsOld == np.array([clPointLst[j+1]]))[0],axis=0)
                        allCells = np.append(allCells,np.where(cellsOld == np.array([clPointLst[j]]))[0][:],axis=0)
                    allCells = np.unique(allCells)
                    # print('Found cells to interpolate from: %s'%str(allCells))
                    # -- find center of the current cell 
                    centroidCell = centroidTriangle(closestPoint[0,:], closestPoint[1,:], closestPoint[2,:])
                    # print('ClosestPoint %s, centroid %s' %(str(closestPoint), str(centroidCell)))
                    norms = np.zeros((allCells.shape))
                    for j in range(allCells.shape[0]):
                        targetCell = target.GetCell(allCells[j]).GetPoints()
                        targetCellPts = np.array(targetCell.GetPoint(0)), \
                                        np.array(targetCell.GetPoint(1)), \
                                        np.array(targetCell.GetPoint(2))
                        targetCentroid = centroidTriangle(*targetCellPts)
                        norms[j] = np.linalg.norm(targetCentroid-centroidCell)
                    norms = norms / np.linalg.norm(norms)
                    norms = norms / np.sum(norms)
                    for j in range(allCells.shape[0]): 
                        Wf[i, allCells[j]] = norms[j]
            else:
                print('Cell is not triangle.')
        print('Saving weighting matrix with shape %s'%str(Wf.shape))
        np.save('%s/Wf_%s_folding.npy' %(self.outDir, plochaName.split('.vtk')[0]),Wf)
        return Wf

    # -- function to run symmetric and anytisymetric POD
    def UsymUAsym(self, UBox, plochaName = "plochaHor.vtk",onlyXY=False,indFromFile=False):
        self.loadVTKandTimeLst(plochaName = plochaName)
        if indFromFile:
            Wf = np.load('%s/Wf_%s_folding.npy' %(self.outDir, plochaName.split('.vtk')[0]))
            print('Loading weighting matrix with shape %s'%str(Wf.shape))
        else:
            Wf = self.loadIndicesForFlipping(plochaName = plochaName)
        print('Norm of the weighting matrix %g'%(np.linalg.norm(Wf)))
        sWf = sparse.csr_matrix(Wf)
        USym = np.zeros((UBox.shape))
        UASym = np.zeros((UBox.shape))
        # UFl = np.zeros((UBox.shape))
        for colI in range(UBox.shape[1]):
            if onlyXY:
                UTu = UBox[:,colI].reshape((-1,2))
                UFlipped = sWf.dot(UTu)
                USym[:, colI] = ((UTu + np.array([1,-1]) * UFlipped)/2).reshape(-1)
                UASym[:, colI] = ((UTu + np.array([-1,1]) * UFlipped)/2).reshape(-1)
            else:
                UTu = UBox[:,colI].reshape((-1,3))
                UFlipped = sWf.dot(UTu)
                USym[:, colI] = ((UTu + np.array([1,-1,1]) * UFlipped)/2).reshape(-1)
                UASym[:, colI] = ((UTu + np.array([-1,1,-1]) * UFlipped)/2).reshape(-1)
            
        return USym, UASym #, UFl
    
    # -- function to read field from vtk
    def loadNumpyFromVtk(self,vtkName):
        # -- prepare vtk reader
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(vtkName)
        reader.Update()

        # Get vtkPolyData object from reader
        polydata = reader.GetOutput()

        # Get cell data array from polydata
        fields = []
        for fieldI in range(len(self.procFields)):
            fields.append(np.array(polydata.GetCellData().GetArray(self.procFields[fieldI])))
        
        return fields
    
    # -- function to write vtk from numpy fields
    # -- name -- name of the file, fields -- numpy fields to write, template -- template vtk, dest -- directory to store
    def writeVtkFromNumpy(self,name,fields,template,dest):
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(template)
        reader.Update()

        if not os.path.exists(dest):
            os.makedirs(dest)

        data = reader.GetOutput()
        
        for fieldI in range(len(self.procFields)):
            # Get the array you want to modify
            array = data.GetCellData().GetArray(self.procFields[fieldI])

            # Modify the array
            if array.GetNumberOfComponents() == 1:
                for i in range(array.GetNumberOfTuples()):
                    # print(i,fields[fieldI].shape)
                    array.SetValue(i, fields[fieldI][i])
            else:
                for i in range(array.GetNumberOfTuples()):
                    array.SetTuple(i, fields[fieldI][i])
                
                # array.InsertTuple(i, fields[fieldI][i])

        print('Writing file %s/%s' %(dest,name))
        writer = vtk.vtkGenericDataObjectWriter()
        writer.SetFileName('%s/%s'%(dest,name))
        # writer.SetInputData(reader.GetOutput())
        writer.SetInputData(data)
        writer.Write()
    
    # -- function to reconstruct vtks together
    # -- destName -- what to merge (must be in each processor)
    def reconstructVTKs(self, destName,name='reconstructAVG.vtk',):
        dirLst = os.listdir(self.caseDir)
        procLst = sorted([proc for proc in dirLst if 'processor' in proc])
        reader = vtk.vtkGenericDataObjectReader()
        append = vtk.vtkAppendFilter()
        for proc in procLst:
            reader.SetFileName('%s/%s/%s'%(self.caseDir,proc,destName))
            reader.Update()
            polydata = vtk.vtkUnstructuredGrid()
            polydata.ShallowCopy(reader.GetOutput())
            append.AddInputData(polydata)
        
        append.Update()    

        writer = vtk.vtkGenericDataObjectWriter()
        writer.SetFileName('%s/%s'%(self.outDir,name))
        writer.SetInputData(append.GetOutput())
        writer.Write()

    # --- function to calculate field average
    def calcAvgY(self, avgName = 'avgs'):
        if self.storage == 'case' or self.storage == 'PPS' or self.storage == 'VTK':
            for i in range(len(self.Ys)):
                UBox = np.copy(self.Ys[i])
                UBoxAvg= np.average(UBox,axis=1)
                self.avgs.append(UBoxAvg)
        elif self.storage == 'VTK-parallel':
            dirLst = os.listdir(self.caseDir)
            procLst = sorted([proc for proc in dirLst if 'processor' in proc])
            procNums = [int(proc.split('processor')[1]) for proc in procLst]
            if not self.parallel:
                for procNum in procNums:
                    self.avgOnProc(procNum,avgName=avgName)
            else:
                with Pool(len(procNums)) as p:
                    p.map(self.avgOnProc, procNums)
    
    # -- function to average on processor
    def avgOnProc(self, procNum, avgName = 'avgs'):
        print('Averiging fields from processor %d'%procNum)
        proc = 'processor%d'%procNum
        vtkNazevTu = self.vtkLst[0].replace('0_','%d_'%procNum)
        avgs = self.loadNumpyFromVtk('%s/%s/VTK/%s'%(self.caseDir,proc,vtkNazevTu))
        nTimes = 0
        for vtkI in range(len(self.vtkLst)-1):
            vtkNazevTu = self.vtkLst[vtkI].replace('0_','%d_'%procNum)
            print('Adding file %s' %vtkNazevTu)
            add = self.loadNumpyFromVtk('%s/%s/VTK/%s'%(self.caseDir,proc,vtkNazevTu))
            for fieldI in range(len(self.procFields)):
                avgs[fieldI] += add[fieldI]
            nTimes += 1
        for fieldI in range(len(self.procFields)):
            avgs[fieldI] = avgs[fieldI] / nTimes
        self.writeVtkFromNumpy('avg.vtk', avgs, '%s/%s/VTK/%s'%(self.caseDir,proc,vtkNazevTu), '%s/%s/%s/'%(self.caseDir,proc,avgName))
    
    # -- function to run POD
    def POD(self, UBoxTu):
        print('Running POD, processed matrix size is: (%d,%d)'%UBoxTu.shape)

        PsiBox,sBox,_ = np.linalg.svd(UBoxTu, full_matrices=False)   
        _ = None     
        etaMat   = (PsiBox.T).dot(UBoxTu)
        
        return PsiBox, sBox, etaMat
    
    # -- write and vizualize singular values 
    def writeSingVals(self, singVals, outDir, timeSample, name='singVals'):
        if not os.path.exists('%s/%s'%(outDir,str(timeSample))):             
            os.makedirs('%s/%s'%(outDir,str(timeSample)))
        with open('%s/%s/%s.dat'%(outDir,str(timeSample),name),'w') as fl:
            fl.writelines('x\tsingVal\tsingValLomSumasingVal\tsingValsqLomSumasingValsq\n')
            for j in range(len(singVals)):
                fl.writelines('%d\t%g\t%g\t%g\n'%(j,singVals[j],singVals[j]/np.sum(singVals),singVals[j]**2/np.sum(singVals**2)))
        plt.plot(singVals**2/np.sum(singVals**2),label='%s'%str(timeSample))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(10e-5,1)
        plt.legend()
        plt.savefig('%s/%s/%s.png'%(outDir,str(timeSample),name))
        plt.close()
    
    # -- write and vizualize chronos spectra 
    def writeChronosSpectra(self, chronos, outDir, timeSample, nModes=60):
        print(chronos.shape)
        # for i in range(chronos.shape[1]):
            

    # -- NOTETH: legacy  
    # --- function to run PO decomposition
    def PODlegacy(self, singValsFile = ''):
        for i in range(len(self.Ys)):
            UBox = np.copy(self.Ys[i])
            for colInd in range(UBox.shape[-1]):
                UBox[:,colInd] = UBox[:,colInd] - self.avgs[i]

            print('Running POD, processed matrix size is: (%d,%d)'%UBox.shape)

            PsiBox,sBox,_ = np.linalg.svd(UBox, full_matrices=False)   
            _ = None     
            etaMat   = (PsiBox.T).dot(UBox)

            self.modes.append(PsiBox)
            self.singVals.append(sBox)
            self.chronos.append(etaMat)
            # print(etaMat[0::4,0::4])
            if not singValsFile == '':
                with open(singValsFile+'_'+self.procFields[i]+'.dat','w') as fl:
                    fl.writelines('x\tsingVal\tsingValLomSumasingVal\tsingValsqLomSumasingValsq\n')
                    for j in range(len(self.singVals[-1])):
                        fl.writelines('%d\t%g\t%g\t%g\n'%(j,self.singVals[-1][j],self.singVals[-1][j]/np.sum(self.singVals[-1]),self.singVals[-1][j]**2/np.sum(self.singVals[-1]**2)))

    # --- function to run randomized PO decomposion
    def rPODlegacy(self,rank = 30, pwr=3,singValsFile = '',comp=True):
        for i in range(len(self.Ys)):
            A = np.copy(self.Ys[i])
            for colInd in range(A.shape[-1]):
                A[:,colInd] = A[:,colInd] - self.avgs[i]

            omega = np.random.randn(A.shape[1], rank)
            Y = A.dot(omega)
            for j in range(pwr):
                Y = A.dot((A.T.dot(Y)))
            Q, _ = np.linalg.qr(Y)
            B = Q.T.dot(A)
            print('Running randomized POD, processed matrix size is: (%d,%d)'%B.shape)
            PsiBox,sBox,_ = np.linalg.svd(B, full_matrices=False)   
            _ = None
            PsiBox = Q.dot(PsiBox)

            self.randmodes.append(PsiBox)
            self.randsingVals.append(sBox)
            # self.randchronos.append(etaMat)
            if not singValsFile == '':
                with open(singValsFile+'_'+self.procFields[i]+'.dat','w') as fl:
                    if comp:
                        fl.writelines('x\tsingVal\tsingValLomSumasingVal\tsingValsqLomSumasingValsq\tabsRelsingvalMsingvalRND\n')
                        for j in range(len(self.randsingVals[-1])):
                            fl.writelines('%d\t%g\t%g\t%g\t%g\n'%(j,self.randsingVals[-1][j],self.randsingVals[-1][j]/np.sum(self.randsingVals[-1]),self.randsingVals[-1][j]**2/np.sum(self.randsingVals[-1]**2),np.abs(self.singVals[i][j]-self.randsingVals[-1][j])/self.singVals[i][j]))

    # vizualize etas
    def vizChronos(self,nChronos = 1,myRange = [0,-1],svFig=''):
        for i in range(len(self.procFields)):
            for j in range(nChronos):
                plt.plot(self.timeLst[myRange[0]:myRange[1]],self.chronos[i][j,myRange[0]:myRange[1]],'x')
                plt.title('Chronos %d of the field %s'%(j,self.procFields[i]))
                plt.xlabel('t (s)')
                plt.ylabel('eta%d'%j)
                if not svFig == '':
                    plt.savefig('%s/%s_%d_%s.png'%(self.outDir,svFig,j,self.procFields[i]))
                    plt.close()
                else:
                    plt.show()
    
    def vizSingVals(self,nSingVals = 1,svFig = ''):
        for i in range(len(self.procFields)):
            x = np.arange(1,nSingVals,1)
            plt.plot(x,self.singVals[i][0:len(x)],'x')
            plt.title('Singular values of the field %s'%(self.procFields[i]))
            plt.xlabel('k')
            plt.yscale('log')
            plt.ylabel('sing. val')
            if not svFig == '':
                plt.savefig('%s/%s_%s.png'%(self.outDir,svFig,self.procFields[i]))
                plt.close()
            else:
                plt.show()
    
    # vizualize spectra of chronos
    def vizSpectraChronos(self,nChronos=1,svFig='',ylims = (1e-3,10e3)):
        for i in range(len(self.procFields)):
            for j in range(nChronos):
                etaFourier = np.abs(np.fft.fft(self.chronos[i][j,:]).real)
                etaFourier = etaFourier/np.average(etaFourier[0:10])
                xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
                plt.plot(xFourier,etaFourier,label='%d'%j)
                plt.title('Spectra of the chronos %d of the field %s'%(j,self.procFields[i]))
                plt.xlabel('f (Hz)')
                plt.ylabel('spectra eta%d'%j)
                plt.xscale('log')
                plt.yscale('log')
                plt.ylim(ylims)
                plt.xlim((1,200))
                if not svFig == '':
                    plt.legend()
                    plt.savefig('%s/%s_%d_%s.png'%(self.outDir,svFig,j,self.procFields[i]))
                    # plt.close()
                else:
                    plt.show()

    # vizualize sums of the spectra of chronos
    def vizSpectraSumChronos(self,nChronos=10,svFig='',ylims = (1e-1,1e6)):
        for i in range(len(self.procFields)):
            etaFourierLst = np.empty((0,len(self.timeLst)))
            for j in range(nChronos):
                etaFourier = np.abs(np.fft.fft(self.chronos[i][j,:]).real)
                # print(etaFourierLst.shape,etaFourier.shape)
                etaFourierLst = np.append(etaFourierLst,etaFourier.reshape(1,-1),axis=0)
                xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
            plt.plot(xFourier,np.sum(etaFourierLst,axis=0))
            plt.title('Spectra of the chronos %d of the field %s'%(j,self.procFields[i]))
            plt.xlabel('f (Hz)')
            plt.ylabel('spectra eta%d'%j)
            plt.xscale('log')
            plt.yscale('log')
            # plt.ylim(ylims)
            plt.xlim((1,1000))
            # with open('%s/%s_%s/spctraTopSum%d_%d.csv'%(self.outDir,svFig,self.procFields[i],j,nChronos),'w') as fl:
            #     fl.writelines('t,\tetaU\n')
            #     for etaInd in range(len(etaFourier)):
            #         fl.writelines('%g,\t%g\n'%(xFourier[etaInd],np.sum(etaFourierLst,axis=0)))
            if not svFig == '':
                plt.savefig('%s/%s_%d_%s_%d.png'%(self.outDir,svFig,j,self.procFields[i],nChronos))
                plt.close()
            else:
                plt.show()

    # vizualize spectra of all points
    def vizSpectraPts(self,svFig='',ylims = (1e-1,1e6)):
            UBox = np.copy(self.Ys[0])
            lV  = self.lenVecsNCells[0][0]
            numCells = self.lenVecsNCells[0][1]        

            print(lV,numCells)

            UBox = np.reshape(UBox,(numCells,2,len(self.timeLst)))
            UBoxAvg= np.average(UBox[:,0,:],axis=0)
            # UBoxAvg= np.average(UBox,axis=0)
            # print(UBox[:,1,:])
            print(UBoxAvg.shape,UBox.shape)

            plt.close()
            etaFourier = np.abs(np.fft.fft(UBoxAvg).real)
            # etaFourierLst = np.append(etaFourierLst,etaFourier.reshape(1,-1),axis=0)
            xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
            plt.plot(xFourier,etaFourier)
            # plt.plot(xFourier,etaFourier)[:2001]*xFourier**(3),label='-3')
            plt.title('Spectra of sum of pts U.')
            plt.xlabel('f (Hz)')
            plt.ylabel('spectra sum pts')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            # plt.ylim(ylims)
            # plt.xlim((1,200))
            if not svFig == '':
                plt.savefig('%s/%s_U.png'%(self.outDir,svFig))
                plt.close()
            else:
                plt.show()


            # UBox = np.reshape(UBox,(numCells,2,len(self.timeLst)))
            UBoxAvg= np.average(UBox[:,1,:],axis=0)
            # UBoxAvg= np.average(UBox,axis=0)
            # print(UBox[:,1,:])
            print(UBoxAvg.shape,UBox.shape)

            etaFourier = np.abs(np.fft.fft(UBoxAvg).real)
            # etaFourierLst = np.append(etaFourierLst,etaFourier.reshape(1,-1),axis=0)
            xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
            plt.plot(xFourier,etaFourier)
            # plt.plot(xFourier,etaFourier)[:2001]*xFourier**(3),label='-3')
            plt.title('Spectra of sum of pts V.')
            plt.xlabel('f (Hz)')
            plt.ylabel('spectra sum pts')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            # plt.ylim(ylims)
            # plt.xlim((1,200))
            if not svFig == '':
                plt.savefig('%s/%s_V.png'%(self.outDir,svFig))
                plt.close()
            else:
                plt.show()

    # vizualize spectra of probe
    def vizSpectraProbe(self,svFig='',ylims = (1e-1,1e6)):
            probeData = np.zeros((len(self.timeLst),2))
            with open('%s/postProcessing/probes/0/U'%self.caseDir,'r')as fl:
                lines = fl.readlines()
            print(len(lines))
            for j in range(4,len(self.timeLst)):
                # print(lines[j],lines[j].split('(')[0],lines[j].split('(')[0].split(' ')[0])
                probeData[j,:]  = np.array([lines[j].split('(')[1].split(' ')[0],lines[j].split('(')[1].split(' ')[2].replace(')','')]).astype(float)

            plt.plot(probeData[:,0])
            plt.savefig('%s/probeData_U.png'%(self.outDir))
            plt.close()
            plt.plot(probeData[:,1])
            plt.savefig('%s/probeData_V.png'%(self.outDir))
            plt.close()

            etaFourier = np.abs(np.fft.fft(probeData[:,0]).real)
            # etaFourierLst = np.append(etaFourierLst,etaFourier.reshape(1,-1),axis=0)
            xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
            plt.plot(xFourier,self.smooth(etaFourier)[0:len(xFourier)])


            # plt.plot(xFourier,etaFourier)[:2001]*xFourier**(3),label='-3')
            plt.title('Spectra in probe point 1.')
            plt.xlabel('f (Hz)')
            plt.ylabel('spectra sum pts')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            # plt.ylim(ylims)
            # plt.xlim((1,200))
            if not svFig == '':
                plt.savefig('%s/%s_%s_U.png'%(self.outDir,svFig,self.procFields[0]))
                plt.close()
            else:
                plt.show()

            etaFourier = np.abs(np.fft.fft(probeData[:,1]).real)
            # etaFourierLst = np.append(etaFourierLst,etaFourier.reshape(1,-1),axis=0)
            xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
            plt.plot(xFourier,etaFourier)
            # plt.plot(xFourier,etaFourier)[:2001]*xFourier**(3),label='-3')
            plt.title('Spectra in probe point 1.')
            plt.xlabel('f (Hz)')
            plt.ylabel('spectra sum pts')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            # plt.ylim(ylims)
            # plt.xlim((1,200))
            if not svFig == '':
                plt.savefig('%s/%s_%s_V.png'%(self.outDir,svFig,self.procFields[0]))
                plt.close()
            else:
                plt.show()



            with open('%s/probe1_V.dat'%(self.outDir),'w') as fl:
                strHere = 'f\tspct\n'

                fl.writelines(strHere)
                for etaInd in range(len(etaFourier)):
                    strHere = '%g'%xFourier[etaInd]
                    strHere += '\t%g\n'%etaFourier[etaInd]
                    fl.writelines(strHere)

    # smooth for spectra
    def smooth(self,x, window_len=8, window='hanning'):
        # if x.ndim != 1:
        #     raise ValueError, "smooth only accepts 1 dimension arrays."

        # if x.size < window_len:
        #     raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        # if not window in ['flat', 'hanning', 'hamming', 'bartlett',
                        # 'blackman']:
            # raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    # -- write Spectra
    def writeSpectraChronos(self,nChronos=1,svFig='',ylims = (1e-5,10)):
        for i in range(len(self.procFields)):
            spectraAll = np.zeros((len(self.chronos[i][0,:]),nChronos))
            for j in range(nChronos):
                etaFourier = np.abs(np.fft.fft(self.chronos[i][j,:]).real)
                # etaFourier = self.smooth(etaFourier)
                etaFourier = etaFourier
                xFourier = np.linspace(0,1.0/((self.timeLst[1]-self.timeLst[0])),len(etaFourier))
                spectraAll[:,j] = etaFourier
            if not os.path.exists('%s/%s_%s/'%(self.outDir,svFig,self.procFields[i])):
                os.makedirs('%s/%s_%s/'%(self.outDir,svFig,self.procFields[i]))
            with open('%s/scaledEtas0-%dRe115.dat'%(self.outDir,nChronos),'w') as fl:
                strHere = 'f'
                for j in range(nChronos):
                    strHere += '\teta%d'%j
                    plt.plot(xFourier,spectraAll[:,j],label='%d'%j)
                strHere += '\tsumEta\n'
                fl.writelines(strHere)
                for etaInd in range(len(etaFourier)):
                    strHere = '%g'%xFourier[etaInd]
                    for j in range(nChronos):
                        strHere += '\t%g'%spectraAll[etaInd,j]
                    strHere += '\t%g\n'%np.sum(spectraAll[etaInd,:])
                    fl.writelines(strHere)
            plt.xscale('log')
            plt.yscale('log')        
            plt.legend()
            plt.savefig('%s/scaledEtas0-%dRe115.png'%(self.outDir,nChronos))
            
    # -- write chronos
    def writeChronos(self,nChronos=1,svFig='',ylims = (1e-5,10)):
        for i in range(len(self.procFields)):
            for j in range(nChronos):
                # print(self.chronos[i][j,:])
                if not os.path.exists('%s/%s_%s/'%(self.outDir,svFig,self.procFields[i])):
                    os.makedirs('%s/%s_%s/'%(self.outDir,svFig,self.procFields[i]))
            with open('%s/etas-%dRe115.dat'%(self.outDir,nChronos),'w') as fl:
                strHere = 't'
                for j in range(nChronos):
                    strHere += '\teta%d'%j
                strHere += '\n'
                fl.writelines(strHere)
                for etaInd in range(len(self.chronos[i][j,:])):
                    strHere = '%g'%(0.015*etaInd)
                    for j in range(nChronos):
                        strHere += '\t%g'%self.chronos[i][j,etaInd]
                    strHere += '\n'
                    fl.writelines(strHere)
    
    # -- viz numpy according to template vtk 
    def vizNpVecInTmplVTK(self, fldToViz, tmplVtk, outFileName, nameOfTheField='poleKViz',componentToViz=0):
        print('Saving vtkfield with shape %s into png'%(str(fldToViz.shape)))
        # Read template VTK file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(tmplVtk)
        reader.Update()
        
        tmpl = reader.GetOutput()
        
        # Create the filter with array
        arrayFilter = vtk.vtkDoubleArray()
        arrayFilter.SetName(nameOfTheField)
        arrayFilter.SetNumberOfComponents(1)
        # arrayFilter.SetNumberOfComponents(3)
        arrayFilter.SetNumberOfTuples(tmpl.GetNumberOfCells())

        for i in range(tmpl.GetNumberOfCells()):
            # arrayFilter.SetTuple(i, fldToViz[i])
            arrayFilter.SetTuple1(i, fldToViz[i,componentToViz])
        
        # Add the the filter to data
        tmpl.GetCellData().AddArray(arrayFilter)
        
        # Create a lookup table for colors
        colorLookupTable = vtk.vtkLookupTable()
        colorLookupTable.SetNumberOfColors(256)
        colorLookupTable.SetHueRange(0.0, 0.667)  # Adjust the range for desired colors
        colorLookupTable.Build()

        # Create a mapper for vorticity visualization
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        scalarMin = min(fldToViz[i][componentToViz] for i in range(tmpl.GetNumberOfCells()))
        scalarMax = max(fldToViz[i][componentToViz] for i in range(tmpl.GetNumberOfCells()))

        mapper.SetScalarRange(scalarMin, scalarMax)
        # mapper.SetScalarRange(arrayFilter.GetRange())
        mapper.SetLookupTable(colorLookupTable)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray(nameOfTheField)
        
        # Create an actor for vorticity visualization
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Create a renderer, set background color, and add the actor
        renderer = vtk.vtkRenderer()
        # renderer.SetBackground(0.2, 0.3, 0.4)
        renderer.AddActor(actor)

        # Create a render window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(800, 600)

        # Use vtkWindowToImageFilter to capture the rendered image
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()

        # Save the image as a PNG file
        writer = vtk.vtkPNGWriter()
        writer.SetFileName('%s/%s'%(self.outDir,outFileName))
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()
        
        writer = vtk.vtkGenericDataObjectWriter()
        writer.SetFileName('%s/%s.vtk'%(self.outDir,outFileName.split('.png')[0]))
        # writer.SetInputData(reader.GetOutput())
        writer.SetInputData(tmpl)
        writer.Write()
      
    # --- function to read oF file and to return the length of vector and number of cells -- case
    def getVecLengthCase(self,data):
        for i in range(len(data)):
            if data[i].find('internalField') == 0:                          #it is at the line beginning
                numCells = int(data[i+1])                                   #get number of cells in the mesh
                try:
                    start  = data[i+3].rindex( '(' ) + len( '(' )
                    end    = data[i+3].index( ')', start )
                    auxVar =  data[i+3][start:end]
                    return len([float(numStr) for numStr in auxVar.split(' ')]),numCells
                except ValueError:
                    return [1,numCells]

    # --- function to read oF file and to return the length of vector and number of cells -- PPS
    def getVecLengthPPS(self,data):
        for i in range(len(data)):
            if data[i].find('CELL_DATA') == 0:                          #it is at the line beginning
                return int(data[i+2].replace('\n','').split(' ')[1]), int(data[i+2].replace('\n','').split(' ')[2])

    # --- function to read vol(Scalar/Vector)Field from an oF result file
    def readInternalField(self,data):
        lV  = self.lenVecsNCells[-1][0]
        numCells = self.lenVecsNCells[-1][1]
        for i in range(len(data)):
            if data[i].find('internalField') == 0 and self.storage == 'case' or data[i].find('CELL_DATA') == 0:     #it is at the line beginning
                # numCells = int(data[i-2].replace('\n','').split(' ')[1])                                                  #get number of cells in the mesh
                # auxStr = " ".join(data[i+3:i+4+numCells]).translate(string.maketrans('', ''), '()\n').split(' ')
                auxStr = " ".join(data[i+3:i+4+numCells]).replace(')','').replace('(','').replace('\n','').split(' ')
                return np.array([float(numStr) for numStr in auxStr if isFloat(numStr)]).reshape(numCells,lV)

    # --- function to load openfoam data as case option
    def loadFieldFilesCase(self,field,mkDel = []):
        with open('%s/%g/%s'%(self.caseDir,self.timeLst[0],field), 'r') as file:
            data = file.readlines()

        # -- get the output dimensions
        mkDelTu = mkDel[:]
        self.lenVecsNCells.append(self.getVecLengthCase(data))
        nCells = self.lenVecsNCells[-1][1]
        lenVec = self.lenVecsNCells[-1][0]
        print('Field %s has %d dimensions in %d cells.' %(field,lenVec,nCells))
        nTimes = len(self.timeLst)
        
        # -- allocate the output variable
        Y   = np.zeros((nCells,lenVec,nTimes))
        
        # -- load the data
        for i in range(nTimes):
            print('Reading field %s from time %g'%(field,self.timeLst[i]))
            Y[:,:,i] = self.readInternalField(data)                              #I have the first data preloaded
            if i+1 < nTimes:                                                #if there is another file, load it
                with open('%s/%g/%s'%(self.caseDir,self.timeLst[i+1],field), 'r') as file:
                    data = file.readlines()

        # -- check for empty dimensions in the data
        valsNorm = [np.linalg.norm(Y[:,i,:]) for i in range(lenVec)]
        meanNorm = np.mean(valsNorm)
        for i in range(lenVec):
            if valsNorm[i] <= meanNorm/100:
                mkDelTu.append(i)    

        self.mkDels.append(mkDelTu)

        # -- delete empty dimensions and save the rest
        Y = np.delete(Y,mkDelTu,1)
        Y = np.reshape(Y, (Y.shape[0]*Y.shape[1],nTimes))
        print('I have loaded matrix Y with dimensions %d x %d of the field %s, %s dimension was deleted.'%(Y.shape[0],Y.shape[1],field,mkDelTu))
        self.Ys.append(Y)
        # return np.delete(Y,mkDel,1),lenVec,mkDel
    
    # --- function to load openfoam data as case PPS option
    def loadFieldFilesPPS(self,field,plochaName,mkDel = [],onlyXY=False):
        with open('%s/postProcessing/sample/%g/%s_%s'%(self.caseDir,self.timeLst[0],field,plochaName), 'r') as file:
            data = file.readlines()

        # -- get the output dimensions
        mkDelTu = mkDel[:]
        self.lenVecsNCells.append(self.getVecLengthPPS(data))
        nCells = self.lenVecsNCells[-1][1]
        lenVec = self.lenVecsNCells[-1][0]
        print('Field %s has %d dimensions in %d cells.' %(field,lenVec,nCells))
        nTimes = len(self.timeLst)
        
        # -- allocate the output variable
        Y   = np.zeros((nCells,lenVec,nTimes))
        
        # -- load the data
        for i in range(nTimes):
            print('Reading field %s from time %g'%(field,self.timeLst[i]))
            Y[:,:,i] = self.readInternalField(data)                              #I have the first data preloaded
            if i+1 < nTimes:                                                #if there is another file, load it
                # if mista == None:
                #     with open('%s/postProcessing/sample/%g/%s_%s'%(self.caseDir,self.timeLst[i+1],field,plochaName), 'r') as file:
                #         data = file.readlines()
                # else:
                    # kam = ('%s/postProcessing/sample/%1.'+str(mista)+'f/%s_%s')
                    # print(self.timeLst[i+1])
                with open('%s/postProcessing/sample/%s/%s_%s'%(self.caseDir,self.timeLstStr[i+1],field,plochaName), 'r') as file:
                    data = file.readlines()

        # -- check for empty dimensions in the data
        valsNorm = [np.linalg.norm(Y[:,i,:]) for i in range(lenVec)]
        meanNorm = np.mean(valsNorm)
        for i in range(lenVec):
            if valsNorm[i] <= meanNorm/100:
                mkDelTu.append(i)    

        if onlyXY and lenVec > 2:
            if "Ver" in plochaName:
                mkDelTu.append(1)
            mkDelTu.append(2)

        self.mkDels.append(mkDelTu)        

        # -- delete empty dimensions and save the rest
        Y = np.delete(Y,mkDelTu,1)
        Y = np.reshape(Y, (Y.shape[0]*Y.shape[1],nTimes))
        print('I have loaded matrix Y with dimensions %d x %d of the field %s, %s dimension was deleted.'%(Y.shape[0],Y.shape[1],field,mkDelTu))
        self.Ys.append(Y)
    
    # --- function to write fields (at the moment 'case' option works)
    def writeField(self, field, templateFieldName, fieldName, caseDir, outDir='100',data=[], plochaName = 'plochaHor.vtk'):
        if self.storage == 'case':
            if not os.path.exists('%s'%(outDir)):
                os.makedirs('%s'%(outDir))
            
            # -- input field properties
            procFldInd = self.procFields.index(templateFieldName)
            mkDel = self.mkDels[procFldInd]
            nCells = self.lenVecsNCells[procFldInd][1]
            lenVec = self.lenVecsNCells[procFldInd][0]
        
            # -- replace location and object in out file
            if data == []:
                with open('%s/%g/%s'%(caseDir, self.timeLst[0], templateFieldName), 'r') as file:
                    data = file.readlines()
                    idStr = ['location', 'object']
                    pVals = [outDir.split('/')[-2],fieldName]
                    
                    for j in range(len(idStr)):
                        for k in range(len(data)):
                            fInd = data[k].find(idStr[j])
                            if fInd>-1:
                                data[k] = data[k][:fInd] + idStr[j] + '\t' + pVals[j] + ';\n'
                                break
            
            print('I am writing field %s with %d cells and %d length of vector.'%(fieldName,nCells,lenVec))
            field = np.reshape(field,(nCells,lenVec-len(mkDel)))
            field = np.insert(field,mkDel,0,axis=1)                    
            # -- write in the data
            if field.shape[1] == 1:
                for i in range(len(data)):
                    if data[i].find('internalField') == 0:                          #it is at the line beginning
                        for j in range(3,nCells+3):
                            data[i+j] = '%5.4e\n'%(field[j-3])                  #type will be numpy ndArray
                        break
            else:
                for i in range(len(data)):
                    if data[i].find('internalField') == 0:                          #it is at the line beginning
                        for j in range(3,nCells+3):
                            data[i+j] = '(%5.4e %5.4e %5.4e)\n'%tuple([field[j-3,k] for k in range(lenVec)])
                        break
                
            with open('%s/%s'%(outDir,fieldName), 'w') as file:
                file.writelines( data )
            return data
        elif self.storage == 'PPS':
            if not os.path.exists('%s'%(outDir)):
                os.makedirs('%s'%(outDir))
            
            # -- input field properties
            procFldInd = self.procFields.index(templateFieldName)
            mkDel = self.mkDels[procFldInd]
            nCells = self.lenVecsNCells[procFldInd][1]
            lenVec = self.lenVecsNCells[procFldInd][0]
        
            # -- replace location and object in out file
            if data == []:
                with open(self.vtkFiles[0], 'r') as file:
                    data = file.readlines()
            
            print('I am writing field %s with %d cells and %d length of vector.'%(fieldName,nCells,lenVec))
            field = np.reshape(field,(nCells,lenVec-len(mkDel)))
            field = np.insert(field,mkDel,0,axis=1)                    
            # -- write in the data
            if field.shape[1] == 1:
                for i in range(len(data)):
                    if data[i].find('CELL_DATA') == 0:                          #it is at the line beginning
                        for j in range(3,int(floor(nCells/10))+3):
                            data[i+j] = '%5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e\n'%tuple([field[10*(j-3)+k,0] for k in range(10)])                #type will be numpy ndArray
                        poslR = ''
                        for num in range(nCells%10,0,-1):
                            poslR += '%5.4e '%field[- num,0]
                        poslR += '\n'
                        try:
                            data[i+int(floor(nCells/10))+3] = poslR 
                        except:
                            data.append(poslR)
                        break
            else:
                for i in range(len(data)):
                    if data[i].find('CELL_DATA') == 0:                          #it is at the line beginning
                        for j in range(3,nCells+3):
                            data[i+j] = '%5.4e %5.4e %5.4e\n'%tuple([field[j-3,k] for k in range(lenVec)])
                        break
                
            with open('%s/%s_%s.vtk'%(outDir,fieldName,plochaName), 'w') as file:
                file.writelines( data )
            return data
    
    # -- end of legacy