import os, re, time
import numpy as np
import pandas as pd
import multiprocessing as mp
import utils

output_dir = 'train/'
fsV = 30
AOA = [-5, 0, 5]
airfoil_Lst = os.listdir('airfoil_database/')
cpu_to_use = 4

def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                outFile.write(line)

    if os.system("gmsh airfoil.geo -3 -format msh2 airfoil.msh > /dev/null") != 0:
        print("error during mesh creation!")
        return(-1)
    
    # 把.msh文件中的 $Nodes 與 $EndNodes 間的資料填進 internalProbes 的 points
    node_coordinates = utils.extract_node_coordinates('airfoil.msh')
    points = "".join(f"({x} {y} {z})\n" for x, y, z in node_coordinates)
    with open('system/internalProbes_template', 'r') as template:
        with open('system/internalProbes','w') as file:
            for line in template:
                line = line.replace(r'<points>',points)
                file.write(line)

    if os.system("gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return(-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

    return(0)

def runSim(freestreamX,freestreamY):
    with open("U_template", 'rt') as inFile:
        with open("0/U",'wt') as outFile:
            for line in inFile:
                line = line.replace('VEL_X',str(freestreamX))
                line = line.replace("VEL_Y",str(freestreamY))
                outFile.write(line)

    os.system('simpleFoam > foam.log')
    return 0

def outputProcessing(caseName, freestreamX, freestreamY, dataDir=output_dir, res=512): 
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    file_dir = f'{caseName}/postProcessing/internalProbes/'
    file_lst = os.listdir(file_dir)
    filtered_lst = [x for x in file_lst if x.isdigit()]
    int_lst = sorted([int(item) for item in filtered_lst])
    finalstep = int_lst[-1]
    file_path = file_dir + str(finalstep) + '/points.xy'
    ar = np.loadtxt(file_path)
    df = pd.DataFrame({'x':ar[:,0],'y':ar[:,1],'p':ar[:,3],'u':ar[:,4],'v':ar[:,5]})
    filtered_df = df[((df['x'] < 1.5) & (df['x'] > -0.5)) & ((df['y']<1) & (df['y'] > -1))].sort_values('x')
    # Normalized to x in -1.5~0.5 y in -1~1
    normalized_x = (filtered_df['x'] + 0.5) / 2
    normalized_y = (filtered_df['y'] + 1 ) / 2
    # Fill the output
    npOutput = np.zeros((6,res,res))
    npOutput[0,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = freestreamX
    npOutput[1,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = freestreamY
    npOutput[2,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = 1.0
    npOutput[3,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = filtered_df['p']
    npOutput[4,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = filtered_df['u']
    npOutput[5,(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = filtered_df['v']

    utils.saveAsImage(f'data_pictures/{caseName}_pressure.png', npOutput[3])
    utils.saveAsImage(f'data_pictures/{caseName}velX.png', npOutput[4])
    utils.saveAsImage(f'data_pictures/{caseName}velY.png', npOutput[5])
    utils.saveAsImage(f'data_pictures/{caseName}inputX.png', npOutput[0])
    utils.saveAsImage(f'data_pictures/{caseName}inputY.png', npOutput[1])

    fileName = dataDir + caseName
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, a=npOutput)


def fullprocess(caseName, fsX, fsY):
    startTime_case = time.time()
    os.chdir(f'{caseName}')
    if genMesh(airfoilFile='data0') != 0:
        print('\tMesh generation failed, aborting.')
        os.chdir('..')
        return -1
    print(f'{caseName} : Doing runSim.')
    if runSim(fsX, fsY) != 0:
        print('runSim failed.')
        return -1
    os.chdir('..')
    print(f'{caseName} : runSim done.')
    outputProcessing(caseName, fsX, fsY)
    totalTime = time.time() - startTime_case
    usingTime = np.round((totalTime/60), 2)
    os.system(f"rm -r {caseName}")
    print("%s : " %(time.strftime('%X')) + f'{caseName} is done, using time : {usingTime} min\n')
    return 0

sample_Lst = []
def progress(status):
    sample_Lst.append(status)
    print(f'{sample_Lst}\t{len(sample_Lst)} / {len(AOA) * len(airfoil_Lst)}')

utils.makeDirs( ["./data_pictures", "./train"] )

if __name__ == '__main__':
    pool = mp.Pool(cpu_to_use)
    startTime = time.time()
    for aoa in AOA:
        for airfoil in airfoil_Lst:
            basename = os.path.splitext(os.path.basename(airfoil))[0]
            caseName = f'{basename}_{aoa}'
            angle = (np.pi / 180) * aoa
            fsX = fsV * np.cos(angle)
            fsY = fsV * np.sin(angle)
            if not os.path.exists(f'{caseName}'):
                os.system(f'cp -r OpenFOAM {caseName}')
                os.system(f'cp -r airfoil_database/{airfoil} {caseName}/data0')
                print(f'case {caseName} is created.')
            pool.apply_async(fullprocess, args=(caseName, fsX, fsY), callback=progress)
    pool.close()
    pool.join()
    totalTime = (time.time() - startTime)/60
    print(f'Final time elapsed : {np.floor(totalTime / 60)} hour {np.floor(totalTime % 60)} min.')