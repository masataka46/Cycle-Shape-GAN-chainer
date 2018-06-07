import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
# from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# from chainer.training import extensions
# from PIL import Image
import utility as Utility
from make_datasets import Make_datasets_CityScape
import argparse


def parser():
    parser = argparse.ArgumentParser(description='analyse oyster images')
    parser.add_argument('--batchsize', '-b', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--log_file_name', '-lf', type=str, default='log01', help='log file name')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='epoch')
    parser.add_argument('--base_dir', '-bd', type=str, default='/media/webfarmer/HDCZ-UT/dataset/cityScape/',
                        help='base directory name of data-sets')
    parser.add_argument('--img_dirX', '-idX', type=str, default='data/leftImg8bit/train/01_summer_fall_coud_rain/',
                        help='directory name of data X')
    parser.add_argument('--img_dirY', '-idY', type=str, default='data/leftImg8bit/train/02_summer_fine/',
                        help='directory name of data Y')
    parser.add_argument('--seg_dirX', '-sdX', type=str, default='gtFine/train/01_summer_fall_coud_rain/',
                        help='directory name of data X')
    parser.add_argument('--seg_dirY', '-sdY', type=str, default='gtFine/train/02_summer_fine/',
                        help='directory name of data Y')
    parser.add_argument('--input_image_size', '-iim', type=int, default=256, help='input image size, only 256 or 128')


    return parser.parse_args()
args = parser()


#global variants
BATCH_SIZE = args.batchsize
N_EPOCH = args.epoch
WEIGHT_DECAY = 0.0005
BASE_CHANNEL = 32
IMG_SIZE = args.input_image_size
IMG_SIZE_BE_CROP_W = 512
IMG_SIZE_BE_CROP_H = 256
BASE_DIR = args.base_dir
DIS_LAST_IMG_SIZE = IMG_SIZE // (2**4)
CO_LAMBDA = 10.0
CO_GAMMA = 1.0
OUT_PUT_IMG_NUM = 6
LOG_FILE_NAME = args.log_file_name
CLASS_NUM = 35 #cityScape dataset


keep_prob_rate = 0.5

seed = 1234
np.random.seed(seed=seed)

out_image_dir = './out_images_cycleGAN' #output image file
out_model_dir = './out_models_cycleGAN' #output model file

try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    print("mkdir error")
    pass
# base_dir, img_width, img_height, image_dirX, image_dirY, image_dirX_seg, image_dirY_seg,
                 #img_width_be_crop, img_height_be_crop, crop_flag=False
make_data = Make_datasets_CityScape(BASE_DIR, IMG_SIZE, IMG_SIZE, args.img_dirX, args.img_dirY, args.seg_dirX, args.seg_dirY,
                                    IMG_SIZE_BE_CROP_W, IMG_SIZE_BE_CROP_H, crop_flag=True)
iniW = chainer.initializers.Normal(scale=0.02)

#generator X for image size = 256------------------------------------------------------------------
class GeneratorX2Y_256(chainer.Chain):
    def __init__(self):
        super(GeneratorX2Y_256, self).__init__(
            
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block7
            res7Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res7Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block8
            res8Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res8Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block9
            res9Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res9Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64*2, 64*2), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128*2, 128*2), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Residual Block 7
        r7 = self.res7Conv1(r6)
        r7 = self.bnR7C1(r7)
        r7 = F.relu(r7)
        r7 = self.res7Conv2(r7)
        r7 = self.bnR7C2(r7) + r6
        r7 = F.relu(r7)
        # Residual Block 8
        r8 = self.res8Conv1(r7)
        r8 = self.bnR8C1(r8)
        r8 = F.relu(r8)
        r8 = self.res8Conv2(r8)
        r8 = self.bnR8C2(r8) + r7
        r8 = F.relu(r8)
        # Residual Block 9
        r9 = self.res9Conv1(r8)
        r9 = self.bnR9C1(r9)
        r9 = F.relu(r9)
        r9 = self.res9Conv2(r9)
        r9 = self.bnR9C2(r9) + r8
        r9 = F.relu(r9)
        # Up 1
        h = self.upConv1(r9)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        h = F.tanh(h)

        return h


#generator Y for image size = 256------------------------------------------------------------------
class GeneratorY2X_256(chainer.Chain):
    def __init__(self):
        super(GeneratorY2X_256, self).__init__(
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block7
            res7Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res7Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block8
            res8Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res8Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block9
            res9Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res9Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64*2, 64*2), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128*2, 128*2), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Residual Block 7
        r7 = self.res7Conv1(r6)
        r7 = self.bnR7C1(r7)
        r7 = F.relu(r7)
        r7 = self.res7Conv2(r7)
        r7 = self.bnR7C2(r7) + r6
        r7 = F.relu(r7)
        # Residual Block 8
        r8 = self.res8Conv1(r7)
        r8 = self.bnR8C1(r8)
        r8 = F.relu(r8)
        r8 = self.res8Conv2(r8)
        r8 = self.bnR8C2(r8) + r7
        r8 = F.relu(r8)
        # Residual Block 9
        r9 = self.res9Conv1(r8)
        r9 = self.bnR9C1(r9)
        r9 = F.relu(r9)
        r9 = self.res9Conv2(r9)
        r9 = self.bnR9C2(r9) + r8
        r9 = F.relu(r9)
        # Up 1
        h = self.upConv1(r9)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        h = F.tanh(h)
        # print("h.data.shape 12", h.data.shape)
        return h


#generator X for image size = 128------------------------------------------------------------------
class GeneratorX2Y_128(chainer.Chain):
    def __init__(self):
        super(GeneratorX2Y_128, self).__init__(
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64, 64), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128, 128), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Up 1
        h = self.upConv1(r6)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        h = F.tanh(h)

        return h


#generator Y for image size = 128------------------------------------------------------------------
class GeneratorY2X_128(chainer.Chain):
    def __init__(self):
        super(GeneratorY2X_128, self).__init__(
            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2 = L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, outsize=(64, 64), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1, outsize=(128, 128), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, 3, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            #batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(3),
        )

    def __call__(self, x, train=True):
        #First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        # print("h.data.shape 1", h.data.shape)
        #Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        # print("h.data.shape 2", h.data.shape)
        #Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        # print("h.data.shape3 ", h.data.shape)
        #Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # print("h.data.shape 4", h.data.shape)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # print("h.data.shape 5", h.data.shape)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # print("h.data.shape 6", h.data.shape)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # print("h.data.shape 7", h.data.shape)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # print("h.data.shape 8", h.data.shape)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # print("h.data.shape 9", h.data.shape)
        # Up 1
        h = self.upConv1(r6)
        h = self.bnU1(h)
        h = F.relu(h)
        # print("h.data.shape 10", h.data.shape)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # print("h.data.shape 11", h.data.shape)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        h = F.tanh(h)
        # print("h.data.shape 12", h.data.shape)
        return h


#discriminator X-----------------------------------------------------------------
class DiscriminatorX(chainer.Chain):
    def __init__(self):
        super(DiscriminatorX, self).__init__(
            conv1=L.Convolution2D(3, BASE_CHANNEL * 2, ksize=4, stride=2, pad=1, initialW=iniW),
            conv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=4, stride=2, pad=1, initialW=iniW),
            conv3=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 8, ksize=4, stride=2, pad=1, initialW=iniW),
            conv4=L.Convolution2D(BASE_CHANNEL * 8, BASE_CHANNEL * 16, ksize=4, stride=2, pad=1, initialW=iniW),
            conv5=L.Convolution2D(BASE_CHANNEL * 16, 1, ksize=DIS_LAST_IMG_SIZE, stride=2, pad=0, initialW=iniW),

            # batch normalization
            bnC1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnC2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnC3=L.BatchNormalization(BASE_CHANNEL * 8),
            bnC4=L.BatchNormalization(BASE_CHANNEL * 16),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bnC1(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv2(h)
        h = self.bnC2(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv3(h)
        h = self.bnC3(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv4(h)
        h = self.bnC4(h)
        h = F.leaky_relu(h, slope=0.2)
        # print("h.data.shape", h.data.shape)
        h = self.conv5(h)
        h = F.reshape(h, (-1, 1))
        out = F.sigmoid(h)
        return out


#discriminator Y-----------------------------------------------------------------
class DiscriminatorY(chainer.Chain):
    def __init__(self):
        super(DiscriminatorY, self).__init__(
            conv1=L.Convolution2D(3, BASE_CHANNEL * 2, ksize=4, stride=2, pad=1, initialW=iniW),
            conv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=4, stride=2, pad=1, initialW=iniW),
            conv3=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 8, ksize=4, stride=2, pad=1, initialW=iniW),
            conv4=L.Convolution2D(BASE_CHANNEL * 8, BASE_CHANNEL * 16, ksize=4, stride=2, pad=1, initialW=iniW),
            conv5=L.Convolution2D(BASE_CHANNEL * 16, 1, ksize=DIS_LAST_IMG_SIZE, stride=2, pad=0, initialW=iniW),

            # batch normalization
            bnC1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnC2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnC3=L.BatchNormalization(BASE_CHANNEL * 8),
            bnC4=L.BatchNormalization(BASE_CHANNEL * 16),
        )

    def __call__(self, x, train=True):
        h = self.conv1(x)
        h = self.bnC1(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv2(h)
        h = self.bnC2(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv3(h)
        h = self.bnC3(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv4(h)
        h = self.bnC4(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.conv5(h)
        h = F.reshape(h, (-1, 1))
        out = F.sigmoid(h)
        return out


#Segmentor X-----------------------------------------------------------------
class SegmentorX(chainer.Chain):
    def __init__(self):
        super(SegmentorX, self).__init__(

            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block7
            res7Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res7Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block8
            res8Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res8Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block9
            res9Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res9Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1,
                                      outsize=(64 * 2, 64 * 2), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1,
                                      outsize=(128 * 2, 128 * 2), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, CLASS_NUM, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            # batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(CLASS_NUM),
        )

    def __call__(self, x, train=True):
        # First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        # Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        # Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        # Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Residual Block 7
        r7 = self.res7Conv1(r6)
        r7 = self.bnR7C1(r7)
        r7 = F.relu(r7)
        r7 = self.res7Conv2(r7)
        r7 = self.bnR7C2(r7) + r6
        r7 = F.relu(r7)
        # Residual Block 8
        r8 = self.res8Conv1(r7)
        r8 = self.bnR8C1(r8)
        r8 = F.relu(r8)
        r8 = self.res8Conv2(r8)
        r8 = self.bnR8C2(r8) + r7
        r8 = F.relu(r8)
        # Residual Block 9
        r9 = self.res9Conv1(r8)
        r9 = self.bnR9C1(r9)
        r9 = F.relu(r9)
        r9 = self.res9Conv2(r9)
        r9 = self.bnR9C2(r9) + r8
        r9 = F.relu(r9)
        # Up 1
        h = self.upConv1(r9)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        # h = F.tanh(h)

        return h


#Segmentor Y-----------------------------------------------------------------
class SegmentorY(chainer.Chain):
    def __init__(self):
        super(SegmentorY, self).__init__(

            # First Convolution
            convInit=L.Convolution2D(3, BASE_CHANNEL, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128
            # Down 1
            downConv1=L.Convolution2D(BASE_CHANNEL, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1, initialW=iniW),
            # Down 2
            downConv2=L.Convolution2D(BASE_CHANNEL * 2, BASE_CHANNEL * 4, ksize=3, stride=2, pad=1, initialW=iniW),
            # Residual Block1
            res1Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res1Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block2
            res2Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res2Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block3
            res3Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res3Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block4
            res4Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res4Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block5
            res5Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res5Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block6
            res6Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res6Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block7
            res7Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res7Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block8
            res8Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res8Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Residual Block9
            res9Conv1=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            res9Conv2=L.Convolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 4, ksize=3, stride=1, pad=1, initialW=iniW),
            # Up 1
            upConv1=L.Deconvolution2D(BASE_CHANNEL * 4, BASE_CHANNEL * 2, ksize=3, stride=2, pad=1,
                                      outsize=(64 * 2, 64 * 2), initialW=iniW),
            # Up 2
            upConv2=L.Deconvolution2D(BASE_CHANNEL * 2, BASE_CHANNEL, ksize=3, stride=2, pad=1,
                                      outsize=(128 * 2, 128 * 2), initialW=iniW),
            # Last Convolution
            convLast=L.Convolution2D(BASE_CHANNEL, CLASS_NUM, ksize=7, stride=1, pad=3, initialW=iniW),  # 128x128 to 128x128

            # batch normalization
            bnCI=L.BatchNormalization(BASE_CHANNEL),
            bnD1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnD2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR1C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR2C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR3C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR4C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR5C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR6C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR7C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR8C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C1=L.BatchNormalization(BASE_CHANNEL * 4),
            bnR9C2=L.BatchNormalization(BASE_CHANNEL * 4),
            bnU1=L.BatchNormalization(BASE_CHANNEL * 2),
            bnU2=L.BatchNormalization(BASE_CHANNEL),
            bnCL=L.BatchNormalization(CLASS_NUM),
        )

    def __call__(self, x, train=True):
        # First Convolution
        h = self.convInit(x)
        h = self.bnCI(h)
        h = F.relu(h)
        # Down 1
        h = self.downConv1(h)
        h = self.bnD1(h)
        h = F.relu(h)
        # Down 2
        h = self.downConv2(h)
        h = self.bnD2(h)
        hd2 = F.relu(h)
        # Residual Block 1
        r1 = self.res1Conv1(hd2)
        r1 = self.bnR1C1(r1)
        r1 = F.relu(r1)
        r1 = self.res1Conv2(r1)
        r1 = self.bnR1C2(r1) + hd2
        r1 = F.relu(r1)
        # Residual Block 2
        r2 = self.res2Conv1(r1)
        r2 = self.bnR2C1(r2)
        r2 = F.relu(r2)
        r2 = self.res2Conv2(r2)
        r2 = self.bnR2C2(r2) + r1
        r2 = F.relu(r2)
        # Residual Block 3
        r3 = self.res3Conv1(r2)
        r3 = self.bnR3C1(r3)
        r3 = F.relu(r3)
        r3 = self.res3Conv2(r3)
        r3 = self.bnR3C2(r3) + r2
        r3 = F.relu(r3)
        # Residual Block 4
        r4 = self.res4Conv1(r3)
        r4 = self.bnR4C1(r4)
        r4 = F.relu(r4)
        r4 = self.res4Conv2(r4)
        r4 = self.bnR4C2(r4) + r3
        r4 = F.relu(r4)
        # Residual Block 5
        r5 = self.res5Conv1(r4)
        r5 = self.bnR5C1(r5)
        r5 = F.relu(r5)
        r5 = self.res5Conv2(r5)
        r5 = self.bnR5C2(r5) + r4
        r5 = F.relu(r5)
        # Residual Block 6
        r6 = self.res6Conv1(r5)
        r6 = self.bnR6C1(r6)
        r6 = F.relu(r6)
        r6 = self.res6Conv2(r6)
        r6 = self.bnR6C2(r6) + r5
        r6 = F.relu(r6)
        # Residual Block 7
        r7 = self.res7Conv1(r6)
        r7 = self.bnR7C1(r7)
        r7 = F.relu(r7)
        r7 = self.res7Conv2(r7)
        r7 = self.bnR7C2(r7) + r6
        r7 = F.relu(r7)
        # Residual Block 8
        r8 = self.res8Conv1(r7)
        r8 = self.bnR8C1(r8)
        r8 = F.relu(r8)
        r8 = self.res8Conv2(r8)
        r8 = self.bnR8C2(r8) + r7
        r8 = F.relu(r8)
        # Residual Block 9
        r9 = self.res9Conv1(r8)
        r9 = self.bnR9C1(r9)
        r9 = F.relu(r9)
        r9 = self.res9Conv2(r9)
        r9 = self.bnR9C2(r9) + r8
        r9 = F.relu(r9)
        # Up 1
        h = self.upConv1(r9)
        h = self.bnU1(h)
        h = F.relu(h)
        # Up 2
        h = self.upConv2(h)
        h = self.bnU2(h)
        h = F.relu(h)
        # Last Convolution
        h = self.convLast(h)
        h = self.bnCL(h)
        # h = F.relu(h)
        # h = F.tanh(h)

        return h


if IMG_SIZE == 256:
    genX2Y = GeneratorX2Y_256() # model for input image size = 256
    genY2X = GeneratorY2X_256() # model for input image size = 256
else:
    genX2Y = GeneratorX2Y_128() # model for input image size = 128
    genY2X = GeneratorY2X_128() # model for input image size = 128

disX = DiscriminatorX()
disY = DiscriminatorY()

segX = SegmentorX()
segY = SegmentorY()

genX2Y.to_gpu()
genY2X.to_gpu()
disX.to_gpu()
disY.to_gpu()
segX.to_gpu()
segY.to_gpu()

optimizer_genX2Y = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_disX = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_genY2X = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_disY = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_segX = optimizers.Adam(alpha=0.0002, beta1=0.5)
optimizer_segY = optimizers.Adam(alpha=0.0002, beta1=0.5)


optimizer_genX2Y.setup(genX2Y)
optimizer_disX.setup(disX)
optimizer_genY2X.setup(genY2X)
optimizer_disY.setup(disY)
optimizer_segX.setup(segX)
optimizer_segY.setup(segY)


optimizer_genX2Y.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_disX.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_genY2X.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_disY.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_segX.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))
optimizer_segY.add_hook(chainer.optimizer.WeightDecay(WEIGHT_DECAY))


#training loop
for epoch in range(0, N_EPOCH):
    sum_loss_gen_total = np.float32(0)
    sum_loss_gen_X = np.float32(0)
    sum_loss_gen_Y = np.float32(0)
    sum_loss_dis_total = np.float(0)
    sum_loss_dis_X = np.float32(0)
    sum_loss_dis_Y = np.float32(0)
    sum_loss_cycle_X2Y = np.float32(0)
    sum_loss_cycle_Y2X = np.float32(0)
    sum_loss_seg_X = np.float32(0)
    sum_loss_seg_Y = np.float32(0)
    sum_loss_seg_Y2X = np.float32(0)
    sum_loss_seg_X2Y = np.float32(0)
    sum_loss_seg_total = np.float32(0)


    make_data.make_data_for_1_epoch() #shuffle training data
    len_data = min(make_data.image_fileX_num, make_data.image_fileY_num)

    for i in range(0, len_data, BATCH_SIZE):
        # print("now i =", i)
        imagesX_np, imagesY_np, imagesX_seg_np, imagesY_seg_np = make_data.get_data_for_1_batch(i, BATCH_SIZE)
        # print("imagesX_np.shape", imagesX_np.shape)
        images_X = Variable(cuda.to_gpu(imagesX_np))
        images_Y = Variable(cuda.to_gpu(imagesY_np))
        seg_X = Variable(cuda.to_gpu(imagesX_seg_np))
        seg_Y = Variable(cuda.to_gpu(imagesY_seg_np))

        # stream around generator
        #
        images_X2Y = genX2Y(images_X)
        images_Y2X = genY2X(images_Y)

        #reverse
        images_X2Y2X = genY2X(images_X2Y)
        images_Y2X2Y = genX2Y(images_Y2X)

        #discriminator
        out_dis_X_real = disX(images_X)
        out_dis_Y_real = disY(images_Y)
        out_dis_X_fake = disX(images_Y2X)
        out_dis_Y_fake = disY(images_X2Y)

        #Segmenttor
        out_seg_X = segX(images_X)
        out_seg_Y2X = segX(images_Y2X)
        out_seg_Y = segY(images_Y)
        out_seg_X2Y = segY(images_X2Y)

        #Cycle Consistency Loss
        loss_cycle_X = F.mean(F.absolute_error(images_X, images_X2Y2X))
        loss_cycle_Y = F.mean(F.absolute_error(images_Y, images_Y2X2Y))
        #Adversarial Loss
        # loss_adv_X_dis = F.mean(- F.log(out_dis_X_real) - F.log(1 - out_dis_X_fake))
        # loss_adv_Y_dis = F.mean(- F.log(out_dis_Y_real) - F.log(1 - out_dis_Y_fake))
        # print("np.mean(out_dis_X_fake.data) ", np.mean(out_dis_X_fake.data))
        # loss_adv_X_gen = F.mean(- F.log(out_dis_X_fake))
        # print("loss_adv_X_gen.data, ", loss_adv_X_gen.data)
        # loss_adv_Y_gen = F.mean(- F.log(out_dis_Y_fake))

        #make target for adversarial loss

        tar_1_np = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        tar_0_np = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        tar_1 = Variable(cuda.to_gpu(tar_1_np))
        tar_0 = Variable(cuda.to_gpu(tar_0_np))

        # Adversarial Loss
        loss_adv_X_dis = F.mean_squared_error(out_dis_X_real, tar_1) + F.mean_squared_error(out_dis_X_fake, tar_0)
        loss_adv_Y_dis = F.mean_squared_error(out_dis_Y_real, tar_1) + F.mean_squared_error(out_dis_Y_fake, tar_0)
        loss_adv_X_gen = F.mean_squared_error(out_dis_X_fake, tar_1)
        loss_adv_Y_gen = F.mean_squared_error(out_dis_Y_fake, tar_1)

        #Shape consistency Loss
        loss_seg_X = F.softmax_cross_entropy(out_seg_X, seg_X)
        loss_seg_Y = F.softmax_cross_entropy(out_seg_Y, seg_Y)
        loss_seg_Y2X = F.softmax_cross_entropy(out_seg_Y2X, seg_X)
        loss_seg_X2Y = F.softmax_cross_entropy(out_seg_X2Y, seg_Y)

        #total Loss
        # print("loss_adv_X_gen.data.shape", loss_adv_X_gen.data.shape)
        # print("loss_adv_Y_gen.data.shape", loss_adv_Y_gen.data.shape)
        # print("loss_cycle_X.data.shape", loss_cycle_X.data.shape)
        # print("loss_cycle_Y.data.shape", loss_cycle_Y.data.shape)
        loss_gen_total = loss_adv_X_gen + loss_adv_Y_gen + CO_LAMBDA * (loss_cycle_X + loss_cycle_Y) \
                         + CO_GAMMA * (loss_seg_X + loss_seg_Y + loss_seg_Y2X + loss_seg_X2Y)
        loss_dis_total = loss_adv_X_dis + loss_adv_Y_dis
        loss_seg_total = loss_seg_X + loss_seg_Y + loss_seg_Y2X + loss_seg_X2Y

        # for print
        sum_loss_gen_total += loss_gen_total.data
        sum_loss_gen_X += loss_adv_X_gen.data
        sum_loss_gen_Y += loss_adv_Y_gen.data
        sum_loss_dis_total += loss_dis_total.data
        sum_loss_dis_X += loss_adv_X_dis.data
        sum_loss_dis_Y += loss_adv_Y_dis.data
        sum_loss_cycle_Y2X += loss_cycle_X.data
        sum_loss_cycle_X2Y += loss_cycle_Y.data
        sum_loss_seg_X += loss_seg_X.data
        sum_loss_seg_Y += loss_seg_Y.data
        sum_loss_seg_Y2X += loss_seg_Y2X.data
        sum_loss_seg_X2Y += loss_seg_X2Y.data
        sum_loss_seg_total += loss_seg_total.data

        # print("sum_loss_gen_X", sum_loss_gen_X)
        # # print("sum_loss_gen_Y", sum_loss_gen_Y)
        # # print("sum_loss_dis_X", sum_loss_dis_X)
        # # print("sum_loss_dis_Y", sum_loss_dis_Y)
        # # print("sum_loss_cycle_Y2X", sum_loss_cycle_Y2X)
        # # print("sum_loss_cycle_X2Y", sum_loss_cycle_X2Y)

        # discriminator back prop
        disX.cleargrads()
        disY.cleargrads()
        loss_dis_total.backward()
        optimizer_disX.update()
        optimizer_disY.update()

        # generator back prop
        genX2Y.cleargrads()
        genY2X.cleargrads()
        loss_gen_total.backward()
        optimizer_genX2Y.update()
        optimizer_genY2X.update()

        # segmentor back prop
        segX.cleargrads()
        segY.cleargrads()
        loss_seg_total.backward()
        optimizer_segX.update()
        optimizer_segY.update()



    print("----------------------------------------------------------------------")
    print("epoch =", epoch , ", Total Loss of G =", sum_loss_gen_total / len_data, ", Total Loss of D =", sum_loss_dis_total / len_data,
          "Total Loss of S =", sum_loss_seg_total / len_data)
    print("Discriminator: Loss X =", sum_loss_dis_X / len_data, ", Loss Y =", sum_loss_dis_Y / len_data)
    print("Generator: Loss adv X=", sum_loss_gen_X / len_data, ", Loss adv Y =", sum_loss_gen_Y / len_data,)
    print("Generator: Loss cycle Y2X=", sum_loss_cycle_Y2X / len_data, ", Loss cycle X2Y =", sum_loss_cycle_X2Y / len_data,)
    print("Segmentor: Loss seg X=", sum_loss_seg_X / len_data, ", Loss seg Y =", sum_loss_seg_Y / len_data,
          "Loss seg Y2X=", sum_loss_seg_Y2X / len_data, ", Loss seg X2Y =", sum_loss_seg_X2Y / len_data)


    if epoch % 5 == 0:
        #outupt generated images
        img_X = []
        img_X2Y = []
        img_X2Y2X = []
        img_Y = []
        img_Y2X = []
        img_Y2X2Y = []
        for i in range(OUT_PUT_IMG_NUM):
            imagesX_np, imagesY_np, imagesX_seg, imagesY_seg = make_data.get_data_for_1_batch(i, 1)
            # print("imagesX_np.shape", imagesX_np.shape)

            img_X.append(imagesX_np[0])
            img_Y.append(imagesY_np[0])

            images_X = Variable(cuda.to_gpu(imagesX_np))
            images_Y = Variable(cuda.to_gpu(imagesY_np))

            # stream around generator
            images_X2Y = genX2Y(images_X)
            images_Y2X = genY2X(images_Y)

            img_X2Y.append(images_X2Y.data[0])
            img_Y2X.append(images_Y2X.data[0])

            # reverse
            images_X2Y2X = genY2X(images_X2Y)
            images_Y2X2Y = genX2Y(images_Y2X)

            img_X2Y2X.append(images_X2Y2X.data[0])
            img_Y2X2Y.append(images_Y2X2Y.data[0])

        img_X_np = np.asarray(img_X).transpose((0, 2, 3, 1))
        img_Y_np = np.asarray(img_Y).transpose((0, 2, 3, 1))
        img_X2Y_np = np.asarray(img_X2Y).transpose((0, 2, 3, 1))
        img_Y2X_np = np.asarray(img_Y2X).transpose((0, 2, 3, 1))
        img_X2Y2X_np = np.asarray(img_X2Y2X).transpose((0, 2, 3, 1))
        img_Y2X2Y_np = np.asarray(img_Y2X2Y).transpose((0, 2, 3, 1))

        Utility.make_output_img(img_X_np, img_X2Y_np, img_X2Y2X_np, img_Y_np, img_Y2X_np, img_Y2X2Y_np, out_image_dir,
                                epoch, LOG_FILE_NAME)


