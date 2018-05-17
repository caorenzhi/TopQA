import sys
import os
import re
from os import listdir
from os.path import isfile, join

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


WIDTH = HEIGHT = DEPTH = 52

def main():
    if len(sys.argv) < 5:
        showExample()
        sys.exit(0)

    Resolution = 51        # this is the resolution for our model
    Rotation = 8           # this is the rotation for our model
    PathPdb2Img = sys.argv[1]
    Model = os.path.abspath(sys.argv[2])
    Input = sys.argv[3]
    Output = sys.argv[4]

    if not os.path.isdir(Output):
        os.makedirs(Output)

    IMGFolder = Output + "/IMGFolder"      # this folder is going to store the image information for all input PDB models
    if not os.path.isdir(IMGFolder):
        os.makedirs(IMGFolder)

    ScorePath = Output+"/TopQAScores.txt"
    fh = open(ScorePath,'w')
    model = keras.models.load_model(Model, custom_objects= {'pearson_correlation': pearson_correlation})


    if not os.path.isdir(Input):       # this is a single model
        name = extract_name(Input)
        try:
            cmd = PathPdb2Img+" "+Input +" "+ str(Resolution)+" "+str(Rotation)+ " "+IMGFolder+"/"+name
            os.system(cmd)
            npyOut = GetNPY(join(IMGFolder,name))
            # delete the big image data
            os.system("rm "+join(IMGFolder,name))
            predictedScore = model.predict(npyOut, batch_size=10)
            fh.write(name+"\t"+str(float(predictedScore))+"\n")
        except:
            print("Warning, something wrong when processing "+eachfile+", check "+cmd)
            fh.write(name+"\t0\n")
    else:
        onlyfiles = [f for f in listdir(Input) if isfile(join(Input, f))]
        for eachfile in onlyfiles:
            name = extract_name(eachfile)
            cmd = PathPdb2Img+" "+ join(Input,eachfile) +" "+ str(Resolution)+" "+str(Rotation)+ " "+IMGFolder+"/"+name
            try:
                os.system(cmd)
                npyOut = GetNPY(join(IMGFolder,name))
                # delete the big image data
                os.system("rm "+join(IMGFolder,name))
                #print('x_test shape:', npyOut.shape)
                predictedScore = model.predict(npyOut, batch_size=10)
                fh.write(name+"\t"+str(float(predictedScore))+"\n")
            except:
                print("Warning, something wrong when processing "+eachfile+", check "+cmd)
                fh.write(name+"\t0\n")
            #sys.exit(0)

    os.system("rmdir "+IMGFolder)

    fh.close()


def pearson_correlation(y_true, y_pred):
    """
    Custom keras metric to calculate the correlation between the true and predicted values.
    :param y_true: Tensor of true values
    :param y_pred: Tensor of predicted values
    :return: Tensor of correlation between the true and predicted values
    """
    true_dif = y_true - K.mean(y_true)
    pred_dif = y_pred - K.mean(y_pred)

    numerator = K.sum(true_dif * pred_dif)

    true_denom = K.sqrt(K.sum(K.square(true_dif)))
    pred_denom = K.sqrt(K.sum(K.square(pred_dif)))
    denominator = true_denom * pred_denom

    pearsonr = numerator / denominator
    return pearsonr

"""
This function is going to get the image information in npy format
"""
def GetNPY(input):
    file = open(input, 'r')
    three_d_matrix = np.zeros((WIDTH, HEIGHT, DEPTH))
    for line in file:
        if 'first' in line:
            continue
        elif '#END' not in line:
            # adds points to 3D matrix
            match = re.fullmatch(r'(0\.\d+)[ ,\t]+(\d+)[ ,\t](\d+)[ ,\t](\d+)\n', line)
            #print(match)
            #print(line)
            x_value = int(match.group(2))
            y_value = int(match.group(3))
            z_value = int(match.group(4))
            value = float(match.group(1))
            three_d_matrix[x_value, y_value, z_value] = value
        elif '#END' in line:
            break
    three_d_matrix1 = [np.transpose(three_d_matrix)]
    three_d_matrix1 = np.array(three_d_matrix1)
    return three_d_matrix1

"""
Return the file name from this path
"""
def extract_name(Mypath):
    return Mypath.split('/')[-1]

"""
This is a function to show users some examples to run this tool
"""
def showExample():
    print("This is TopQA, it's going to make predictions for a single model or a folder with several models. You need a program called pdb2img to process PDB model, and a trained model called Model520.01, an output folder")
    print("Dependency: Keras, python3 and h5py library")
    print("For example:")
    print("python3 "+sys.argv[0]+" ../resources/pdb2img ../resources/Model_520.01 ../test/T0859_stage1_true/T0859.pdb ../test/Prediction_singleModel")
    print("python3 "+sys.argv[0]+" ../resources/pdb2img ../resources/Model_520.01 ../test/T0859_stage1_true/stage1 ../test/Prediction_ModelPool")

if __name__ == "__main__":
    # prevent using a GPU since it calls them one at a time
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
