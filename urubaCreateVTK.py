from OpenFoamData import OpenFoamData
import numpy as np
import vtk

buff = 1e-3

# -- hranice
xHr = [0.015/2-buff, 0.105+buff]
yHr = [-30.9e-3, 29.1e-3+buff]

poleToProcess = 'U[m_s]'

# -- load vtm file
vtm_file = "../2D_uruba/urubaTest.vtm"  # Replace with the path to your VTM file
vtm_file = "../2D_uruba/both/out/avg.vtm"  # Replace with the path to your VTM file
reader = vtk.vtkXMLMultiBlockDataReader()
reader.SetFileName(vtm_file)
reader.Update()
block = reader.GetOutput().GetBlock(0)

field = np.array(block.GetPointData().GetArray(poleToProcess))
print(field )

# -- load fields
# cells= np.array(block.GetPolys().GetData()).reshape(block.GetNumberOfCells(),-1)[:,1:]
indices = np.empty(0).astype(int)
for i in range(block.GetNumberOfPoints()):
    centX, centY, centZ = block.GetPoint(i)
    if centX <= xHr[0] or centX >= xHr[1] or centY <= yHr[0] or centY >= yHr[1]:
        field[i] = 0

array = block.GetPointData().GetArray(poleToProcess)

# Modify the array
if array.GetNumberOfComponents() == 1:
    for i in range(array.GetNumberOfTuples()):
        # print(i,fields[fieldI].shape)
        array.SetValue(i, field[i])
else:
    for i in range(array.GetNumberOfTuples()):
        array.SetTuple(i, field[i])

vtk_file_name = "../2D_uruba/both/out/avg.vtk"  # Adjust the naming as needed
writer = vtk.vtkXMLDataSetWriter()
writer.SetFileName(vtk_file_name)
writer.SetInputData(block)
writer.Write()

print("VTK files saved successfully.")