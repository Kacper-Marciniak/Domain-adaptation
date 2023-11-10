import numpy as np
import os
import cv2 as cv
from data.utils import FDA_source_to_target_np, match_histogram
from tqdm import tqdm

def change_style(sPathImgSource: str, sPathImgTarget: str, sPathImgOutput: str|None = None, tTargetShape: tuple|None = None):
    if sPathImgOutput is None: sPathImgOutput = sPathImgSource

    aImgSource = cv.imread(sPathImgSource, cv.IMREAD_UNCHANGED)
    aImgTarget = cv.imread(sPathImgTarget, cv.IMREAD_UNCHANGED)

    if tTargetShape is None:
        tTargetShape = aImgSource.shape
    tTargetShape = (tTargetShape[1], tTargetShape[0])

    aImgSource = cv.resize(aImgSource, tTargetShape, cv.INTER_CUBIC)
    aImgTarget = cv.resize(aImgTarget, tTargetShape, cv.INTER_CUBIC)

    aImgSource = np.array(aImgSource).astype(np.float32)
    aImgTarget = np.array(aImgTarget).astype(np.float32)

    aImgSource = aImgSource.transpose((2, 0, 1))
    aImgTarget = aImgTarget.transpose((2, 0, 1))

    aImgOutput = FDA_source_to_target_np(aImgSource, aImgTarget, L=BETA)

    aImgOutput = aImgOutput.transpose((1,2,0))
    aImgOutput = (aImgOutput - np.min(aImgOutput))/(np.max(aImgOutput)-np.min(aImgOutput))*255.0
    aImgOutput = np.round(aImgOutput).astype(np.uint8)
    cv.imwrite(sPathImgOutput, aImgOutput)

def change_histogram(sPathImgSource: str, sPathImgTarget: str, sPathImgOutput: str|None = None, tTargetShape: tuple|None = None, bForceGrayScale: bool = False):
    if sPathImgOutput is None: sPathImgOutput = sPathImgSource

    aImgSource = cv.imread(sPathImgSource)
    aImgTarget = cv.imread(sPathImgTarget)

    if bForceGrayScale:
        aImgSource = cv.cvtColor(aImgSource, cv.COLOR_BGR2GRAY)
        aImgTarget = cv.cvtColor(aImgTarget, cv.COLOR_BGR2GRAY)

    if tTargetShape is None:
        tTargetShape = aImgSource.shape
    tTargetShape = (tTargetShape[1], tTargetShape[0])

    aImgSource = cv.resize(aImgSource, tTargetShape, cv.INTER_CUBIC)

    aImgOutput = match_histogram(aImgSource, aImgTarget)

    cv.imwrite(sPathImgOutput, aImgOutput)

def generate_new_styles(sPathDataset: str, sPathToTargetSample: str, sPathToSave: str, tImageTargetSize: tuple | None = None):
    lSourceImages = [os.path.join(sPathDataset, sFile).lower() for sFile in os.listdir(sPathDataset) if sFile.lower().split('.')[-1] in ('jpg','png')]

    print(f"Changing style of images in \'{sPathDataset}\' to the style of \'{sPathToTargetSample}\'")

    for sImagePath in tqdm(lSourceImages):
        change_style(
            sImagePath,
            sPathToTargetSample,
            os.path.join(sPathToSave, os.path.basename(sImagePath).split('.')[0]+f"_style_{os.path.basename(sPathToTargetSample.split('.')[0])}."+sImagePath.split('.')[-1]),
            tImageTargetSize
        )

def generate_new_histograms(sPathDataset: str, sPathToTargetSample: str, sPathToSave: str, bForceGrayScale: bool = False, tImageTargetSize: tuple | None = None):
    lSourceImages = [os.path.join(sPathDataset, sFile).lower() for sFile in os.listdir(sPathDataset) if sFile.lower().split('.')[-1] in ('jpg','png')]

    print(f"Changing histogram of images in \'{sPathDataset}\' to the style of \'{sPathToTargetSample}\'")

    for sImagePath in tqdm(lSourceImages):
        change_histogram(
            sImagePath,
            sPathToTargetSample,
            os.path.join(sPathToSave, os.path.basename(sImagePath).split('.')[0]+f"_hist_{os.path.basename(sPathToTargetSample.split('.')[0])}."+sImagePath.split('.')[-1]),
            tImageTargetSize,
            bForceGrayScale
        )

# Image paths
PATH_IMAGES = 'images'
PATH_SAMPLES = 'samples'
PATH_OUT = 'out'

# FDA beta value
BETA = 0.001

# Force grayscale for histogram matching
GRAYSCALE = False

# Target image size
TARGET_SIZE = None

for sSample in os.listdir(PATH_SAMPLES):
    generate_new_styles(PATH_IMAGES, os.path.join(PATH_SAMPLES, sSample), PATH_OUT, TARGET_SIZE)
    generate_new_histograms(PATH_IMAGES, os.path.join(PATH_SAMPLES, sSample), PATH_OUT, GRAYSCALE, TARGET_SIZE)
