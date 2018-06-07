import numpy as np
from PIL import Image
# import utility as Utility
import os
import csv
import random
import utility as util

class Make_datasets_CityScape():
    def __init__(self, base_dir, img_width, img_height, image_dirX, image_dirY, image_dirX_seg, image_dirY_seg,
                 img_width_be_crop, img_height_be_crop, crop_flag=False):

        self.base_dir = base_dir
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_be_crop = img_width_be_crop
        self.img_height_be_crop = img_height_be_crop
        # self.list_train_files = []
        self.dirX = base_dir + image_dirX
        self.dirY = base_dir + image_dirY
        # self.seg_file_extension = seg_file_extension

        self.dirX_seg = base_dir + image_dirX_seg
        self.dirY_seg = base_dir + image_dirY_seg

        self.crop_flag = crop_flag

        self.file_listX = os.listdir(self.dirX)
        self.file_listY = os.listdir(self.dirY)

        self.file_listX.sort()
        self.file_listY.sort()
        #
        # self.cityScape_color_chan = np.array([
        #     [0.0, 0.0, 0.0],#0
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 0.0],
        #     [0.0, 0.0, 70.0],#
        #     [0.0, 0.0, 90.0],#
        #     [0.0, 0.0, 110.0],#
        #     [0.0, 0.0, 142.0],#
        #     [0.0, 0.0, 230.0],#
        #     [0.0, 60.0, 100.0],#10
        #     [0.0, 80.0, 100.0],#
        #     [70.0, 70.0, 70.0],#
        #     [70.0, 130.0, 180.0],#
        #     [81.0, 0.0, 81.0],#
        #     [102.0, 102.0, 156.0],#
        #     [107.0, 142.0, 35.0],#
        #     [111.0, 74.0, 0.0],#
        #     [119.0, 11.0, 32.0],#
        #     [128.0, 64.0, 128.0],#
        #     [150.0, 100.0, 100.0],#20
        #     [150.0, 120.0, 90.0],#
        #     [152.0, 251.0, 152.0],#
        #     [153.0, 153.0, 153.0],#
        #     [180.0, 165.0, 180.0],#
        #     [190.0, 153.0, 153.0],#
        #     [220.0, 20.0, 60.0],#
        #     [220.0, 220.0, 0.0],#
        #     [230.0, 150.0, 140.0],#
        #     [244.0, 35.0, 232.0],#
        #     [250.0, 170.0, 30.0],#
        #     [250.0, 170.0, 160.0],#
        #     [255.0, 0.0, 0.0]#
        #     ], dtype=np.float32
        #     )

        
        # file_listX_seg_be = os.listdir(self.dirX_seg)
        # file_listY_seg_be = os.listdir(self.dirY_seg)
        # file_listX_seg_be.sort()
        # file_listY_seg_be.sort()
        # print("self.file_listX_seg_be[0]", file_listX_seg_be[0])
        # print("self.file_listX_seg_be[1]", file_listX_seg_be[1])
        # print("self.file_listX_seg_be[2]", file_listX_seg_be[2])
        # print("self.file_listX_seg_be[3]", file_listX_seg_be[3])
        # print("self.file_listX_seg_be[4]", file_listX_seg_be[4])
        # print("self.file_listX_seg_be[5]", file_listX_seg_be[5])
        #
        #
        # print("len(self.file_listX_seg_be)", len(file_listX_seg_be))
        #
        # print("self.file_listY_seg_be[0]", file_listY_seg_be[0])
        #
        #
        # self.file_listX_seg = self.get_only_img(file_listX_seg_be, self.seg_file_extension)
        # self.file_listY_seg = self.get_only_img(file_listY_seg_be, self.seg_file_extension)

        self.cityScape_color_chan = np.array([
            [0.0, 0.0, 0.0],#0
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [111.0, 74.0, 0.0],  #
            [81.0, 0.0, 81.0],  #
            [128.0, 64.0, 128.0],  #
            [244.0, 35.0, 232.0],  #
            [250.0, 170.0, 160.0],  #
            [230.0, 150.0, 140.0],  #
            [70.0, 70.0, 70.0],  #
            [102.0, 102.0, 156.0],  #
            [190.0, 153.0, 153.0],  #
            [180.0, 165.0, 180.0],  #
            [150.0, 100.0, 100.0],  #
            [150.0, 120.0, 90.0],  #
            [153.0, 153.0, 153.0],  #
            [153.0, 153.0, 153.0],  #
            [250.0, 170.0, 30.0],  #
            [220.0, 220.0, 0.0],  #
            [107.0, 142.0, 35.0],  #
            [152.0, 251.0, 152.0],  #
            [70.0, 130.0, 180.0],  #
            [220.0, 20.0, 60.0],  #
            [255.0, 0.0, 0.0],  #
            [0.0, 0.0, 142.0],  #
            [0.0, 0.0, 70.0],#
            [0.0, 60.0, 100.0],  #
            [0.0, 0.0, 90.0],#
            [0.0, 0.0, 110.0],#
            [0.0, 80.0, 100.0],  #
            [0.0, 0.0, 230.0],#
            [119.0, 11.0, 32.0],#
            [0.0, 0.0, 142.0]  #
            ], dtype=np.float32
            )

        self.image_fileX_num = len(self.file_listX)
        self.image_fileY_num = len(self.file_listY)
        
        print("self.img_width", self.img_width)
        print("self.img_height", self.img_height)
        print("len(self.file_listX)", len(self.file_listX))
        print("len(self.file_listY)", len(self.file_listY))
        print("self.image_fileX_num", self.image_fileX_num)
        print("self.image_fileY_num", self.image_fileY_num)

        # print("self.file_listX_seg[0]", self.file_listX_seg[0])
        # print("self.file_listX[0]", self.file_listX[0])
        #
        # print("len(self.file_listX_seg)", len(self.file_listX_seg))
        # print("len(self.file_listX)", len(self.file_listX))
        #
        # print("len(self.file_listY_seg)", len(self.file_listY_seg))
        # print("len(self.file_listY)", len(self.file_listY))


    def get_only_img(self, list, extension):
        list_mod = []
        for y in list:
            if (y[-9:] == extension): #only .png
                list_mod.append(y)
        return list_mod


    def read_1_data(self, dir, filename_list, width, height, width_be_crop, height_be_crop, margin_H_batch,
                    margin_W_batch, crop_flag, seg_flag=False):
        images = []
        for num, filename in enumerate(filename_list):
            if seg_flag:
                # ex) bremen_000000_000019_leftImg8bit.png to bremen_000000_000019_gtFine_color.png
                str_base, _ = filename.rsplit("_",1)
                filename_seg = str_base + "_gtFine_labelIds.png"
                pilIn = Image.open(dir + filename_seg)
            else:
                pilIn = Image.open(dir + filename)
            # print("pilIn.size", pilIn.size)
            # print("width", width)
            # print("height", height)
            # print("margin_H_batch", margin_H_batch)
            # print("margin_W_batcW", margin_W_batch)

            if crop_flag:
                pilIn = pilIn.resize((width_be_crop, height_be_crop))
                pilResize = self.crop_img(pilIn, width, height, margin_W_batch[num], margin_H_batch[num])
                # print("pilResize.size", pilResize.size)
            else:
                pilResize = pilIn.resize((width, height))

            # image = np.asarray(pilResize, dtype=np.float32)

            if seg_flag:
                image = np.asarray(pilResize, dtype=np.int32)
                # image = image[:,:,:3]
                # print("state 3.1")
                # image = self.convert_color_to_indexInt(image)
                # print("state 3.2")
                image_t = image
            else:
                image = np.asarray(pilResize, dtype=np.float32)
                image_t = np.transpose(image, (2, 0, 1))
            # except:
            #     print("filename =", filename)
            #     image_t = image.reshape(image.shape[0], image.shape[1], 1)
            #     image_t = np.tile(image_t, (1, 1, 3))
            #     image_t = np.transpose(image_t, (2, 0, 1))
            images.append(image_t)

        return np.asarray(images)


    def normalize_data(self, data):
        data0_2 = data / 127.5
        data_norm = data0_2 - 1.0
        # data_norm = data / 255.0
        # data_norm = data - 1.0
        return data_norm


    def crop_img(self, data, output_img_W, output_img_H, margin_W, margin_H):
        cropped_img = data.crop((margin_W, margin_H, margin_W + output_img_W, margin_H + output_img_H))
        return cropped_img


    def read_1_data_and_convert_RGB(self, dir, filename_list, extension, width, height):
        images = []
        for filename in filename_list:
            pilIn = Image.open(dir + filename[0] + extension).convert('RGB')
            pilResize = pilIn.resize((width, height))
            image = np.asarray(pilResize)
            image_t = np.transpose(image, (2, 0, 1))
            images.append(image_t)
        return np.asarray(images)


    def write_data_to_img(self, dir, np_arrays, extension):

        for num, np_array in enumerate(np_arrays):
            pil_img = Image.fromarray(np_array)
            pil_img.save(dir + 'debug_' + str(num) + extension)


    def convert_color_to_30chan(self, data): # for cityScape dataset when use Tensorflow
        # print("data.shape", data.shape)
        # print("self.cityScape_color_chan.shape", self.cityScape_color_chan.shape)
        d_mod = np.zeros((data.shape[0], data.shape[1], 30), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                # if ele == [0.0, 0.0, 0.0]:
                #     d_mod[h][w][0] = 1.0
                # elif ele == [0.0, 0.0, 70.0]:
                #     d_mod[h][w][1] = 1.0
                # elif ele == [0.0, 0.0, 90.0]:
                #     d_mod[h][w][2] = 1.0
                # elif ele == [0.0, 0.0, 110.0]:
                #     d_mod[h][w][3] = 1.0
                # elif ele == [0.0, 0.0, 142.0]:
                #     d_mod[h][w][4] = 1.0
                # elif ele == [0.0, 0.0, 230.0]:
                #     d_mod[h][w][5] = 1.0
                # elif ele == [0.0, 60.0, 100.0]:
                #     d_mod[h][w][6] = 1.0
                # elif ele == [0.0, 80.0, 100.0]:
                #     d_mod[h][w][7] = 1.0
                # elif ele == [70.0, 70.0, 70.0]:
                #     d_mod[h][w][8] = 1.0
                # elif ele == [70.0, 130.0, 180.0]:
                #     d_mod[h][w][9] = 1.0
                # elif ele == [81.0, 0.0, 81.0]:
                #     d_mod[h][w][10] = 1.0
                # elif ele == [102.0, 102.0, 156.0]:
                #     d_mod[h][w][11] = 1.0
                # elif ele == [107.0, 142.0, 35.0]:
                #     d_mod[h][w][12] = 1.0
                # elif ele == [111.0, 74.0, 0.0]:
                #     d_mod[h][w][13] = 1.0
                # elif ele == [119.0, 11.0, 32.0]:
                #     d_mod[h][w][14] = 1.0
                # elif ele == [128.0, 64.0, 128.0]:
                #     d_mod[h][w][15] = 1.0
                # elif ele == [150.0, 100.0, 100.0]:
                #     d_mod[h][w][16] = 1.0
                # elif ele == [150.0, 120.0, 90.0]:
                #     d_mod[h][w][17] = 1.0
                # elif ele == [152.0, 251.0, 152.0]:
                #     d_mod[h][w][18] = 1.0
                # elif ele == [153.0, 153.0, 153.0]:
                #     d_mod[h][w][19] = 1.0
                # elif ele == [180.0, 165.0, 180.0]:
                #     d_mod[h][w][20] = 1.0
                # elif ele == [190.0, 153.0, 153.0]:
                #     d_mod[h][w][21] = 1.0
                # elif ele == [220.0, 20.0, 60.0]:
                #     d_mod[h][w][22] = 1.0
                # elif ele == [220.0, 220.0, 0.0]:
                #     d_mod[h][w][23] = 1.0
                # elif ele == [230.0, 150.0, 140.0]:
                #     d_mod[h][w][24] = 1.0
                # elif ele == [244.0, 35.0, 232.0]:
                #     d_mod[h][w][25] = 1.0
                # elif ele == [250.0, 170.0, 30.0]:
                #     d_mod[h][w][26] = 1.0
                # elif ele == [250.0, 170.0, 160.0]:
                #     d_mod[h][w][27] = 1.0
                # elif ele == [255.0, 0.0, 0.0]:
                #     d_mod[h][w][28] = 1.0
                # else:
                #     print("else color")
                #     d_mod[h][w][29] = 1.0
                for num, chan in enumerate(self.cityScape_color_chan):
                    # print("ele.shape", ele.shape)
                    # print("chan.shape", chan.shape)
                    if np.allclose(chan, ele):
                        d_mod[h][w][num] = 1.0
        return d_mod

    def convert_30chan_to_color(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(ele):
                    if chan == 1.0:
                        d_mod[h][w] = self.cityScape_color_chan[num]
        return d_mod

    def convert_color_to_indexInt(self, data): # for cityScape dataset when use Chainer
        d_mod = np.zeros((data.shape[0], data.shape[1]), dtype=np.int32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                for num, chan in enumerate(self.cityScape_color_chan):
                    if np.allclose(chan, ele):
                        d_mod[h][w]= num
        return d_mod

    def convert_indexInt_to_color(self, data):
        d_mod = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float32)
        for h, row in enumerate(data):
            for w, ele in enumerate(row):
                d_mod[h][w] = self.cityScape_color_chan[ele]
        return d_mod


    def convert_to_0_1_class_(self, d):
        d_mod = np.zeros((d.shape[0], d.shape[1], d.shape[2], self.class_num), dtype=np.float32)

        for num, image1 in enumerate(d):
            for h, row in enumerate(image1):
                for w, ele in enumerate(row):
                    if int(ele) == 255:#border
                    # if int(ele) == 255 or int(ele) == 0:#border and backgrounds
                        # d_mod[num][h][w][20] = 1.0
                        continue
                    # d_mod[num][h][w][int(ele) - 1] = 1.0
                    d_mod[num][h][w][int(ele)] = 1.0
        return d_mod


    def make_data_for_1_epoch(self):
        # print("len(self.file_listX)", len(self.file_listX))
        # print("self.image_fileX_num", self.image_fileX_num)
        # rand_perX = np.random.permutation(self.image_fileX_num)
        # rand_perY = np.random.permutation(self.image_fileY_num)
        # print("rand_perX", rand_perX)
        # self.image_filesX_1_epoch = self.file_listX[rand_perX]
        # self.image_filesY_1_epoch = self.file_listY[rand_perY]
        self.image_filesX_1_epoch = random.sample(self.file_listX, self.image_fileX_num)
        self.image_filesY_1_epoch = random.sample(self.file_listY, self.image_fileY_num)
        # self.image_filesX_seg_1_epoch = random.sample(self.file_listX_seg, self.image_fileX_num)
        # self.image_filesY_seg_1_epoch = random.sample(self.file_listY_seg, self.image_fileY_num)
        # self.image_filesX_seg_1_epoch = self.file_listX_seg[rand_perX]
        # self.image_filesY_seg_1_epoch = self.file_listY_seg[rand_perY]

        self.margin_H = np.random.randint(0, (self.img_height_be_crop - self.img_height + 1), self.image_fileX_num)
        self.margin_W = np.random.randint(0, (self.img_width_be_crop - self.img_width + 1), self.image_fileY_num)
        # print("self.margin_H", self.margin_H)
        # print("self.margin_W", self.margin_W)


    def get_data_for_1_batch(self, i, batchsize, train_FLAG=True):
        # print("len(self.train_files_1_epoch)", len(self.train_files_1_epoch))
        # if train_FLAG:
        data_batchX = self.image_filesX_1_epoch[i:i + batchsize]
        data_batchY = self.image_filesY_1_epoch[i:i + batchsize]
        # data_batchX_seg = self.image_filesX_seg_1_epoch[i:i + batchsize]
        # data_batchY_seg = self.image_filesY_seg_1_epoch[i:i + batchsize]

        margin_H_batch = self.margin_H[i:i + batchsize]
        margin_W_batch = self.margin_W[i:i + batchsize]

        # else:
        #     print("okasii")
            # data_batch = self.list_val_files[i:i + batchsize]
        imagesX = self.read_1_data(self.dirX, data_batchX, self.img_width, self.img_height, self.img_width_be_crop,
                                   self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        imagesY = self.read_1_data(self.dirY, data_batchY, self.img_width, self.img_height, self.img_width_be_crop,
                                   self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=False)
        # print("state 3")

        imagesX_seg = self.read_1_data(self.dirX_seg, data_batchX, self.img_width, self.img_height,
                self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        imagesY_seg = self.read_1_data(self.dirY_seg, data_batchY, self.img_width, self.img_height,
                self.img_width_be_crop, self.img_height_be_crop, margin_H_batch, margin_W_batch, self.crop_flag, seg_flag=True)
        # print("state 4")

        imagesX_n = self.normalize_data(imagesX)
        imagesY_n = self.normalize_data(imagesY)

        # imagesX_n_seg = self.normalize_data(imagesX_seg)
        # imagesY_n_seg = self.normalize_data(imagesY_seg)

        # labels_0_1 = self.convert_to_0_1_class_(labels)
        return imagesX_n, imagesY_n, imagesX_seg, imagesY_seg

    def make_img_from_label(self, labels, epoch):#labels=(first_number, last_number + 1)
        labels_train = self.train_files_1_epoch[labels[0]:labels[1]]
        labels_val = self.list_val_files[labels[0]:labels[1]]
        labels_train_val = labels_train + labels_val
        labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, labels_train_val, '.png', self.img_width, self.img_height)
        self.write_data_to_img('debug/label_' + str(epoch) + '_',  labels_img_np, '.png')


    def make_img_from_prob(self, probs, epoch):#probs=(data, height, width)..0-20 value
        # print("probs[0]", probs[0])
        print("probs[0].shape", probs[0].shape)
        probs_RGB = util.convert_indexColor_to_RGB(probs)



        # labels_img_np = self.read_1_data_and_convert_RGB(self.SegmentationClass_dir, probs_RGB, '.jpg', self.img_width, self.img_height)
        self.write_data_to_img('debug/prob_' + str(epoch), probs_RGB, '.jpg')


    def get_concat_img_h(self, img1, img2):
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))
        return dst

    def get_concat_img_w(self, img1, img2):
        dst = Image.new('RGB', (img1.width, img1.height + img2.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (0, img1.height))
        return dst


if __name__ == '__main__':
    #debug
    FILE_NAME = 'bremen_000309_000019_gtFine_color.png'
    base_dir = '/media/webfarmer/HDCZ-UT/dataset/cityScape/'
    image_dirX = 'data/leftImg8bit/train/bremen/'
    image_dirY = 'data/leftImg8bit/train/bremen/'
    image_dirX_seg = 'gtFine/train/bremen/'
    image_dirY_seg = 'gtFine/train/bremen/'

    img_width = 200
    img_height = 200

    img_be_crop_width = 400
    img_be_crop_height = 200


    make_datasets_CityScape = Make_datasets_CityScape(base_dir, img_width, img_height, image_dirX, image_dirX,
                                image_dirX_seg, image_dirX_seg, img_be_crop_width, img_be_crop_height, crop_flag=True)

    make_datasets_CityScape.make_data_for_1_epoch()
    imagesX, imagesY, imagesX_seg, imagesY_seg = make_datasets_CityScape.get_data_for_1_batch(0, 1, train_FLAG=True)
    print("imagesX.shape", imagesX.shape)
    print("imagesX.dtype", imagesX.dtype)
    print("imagesX[0][2][10][10]", imagesX[0][2][10][10])

    print("imagesY.shape", imagesY.shape)
    print("imagesY.dtype", imagesY.dtype)
    print("imagesY[0][2][10][10]", imagesY[0][2][10][10])
    print("np.max(imagesY)", np.max(imagesY))
    print("np.min(imagesY)", np.min(imagesY))


    print("imagesX_seg.shape", imagesX_seg.shape)
    print("imagesX_seg.dtype", imagesX_seg.dtype)
    # print("imagesX_seg[0][2][10][10]", imagesX_seg[0][2][10][10])
    print("imagesX_seg[0]", imagesX_seg[0])


    print("imagesY_seg.shape", imagesY_seg.shape)
    print("imagesY_seg.dtype", imagesY_seg.dtype)
    # print("imagesY_seg[0][2][10][10]", imagesY_seg[0][2][10][10])

    # print("self.file_listX_seg[0]", make_datasets_CityScape.file_listX_seg[0])
    print("self.file_listX[0]", make_datasets_CityScape.file_listX[0])

    # print("self.file_listX_seg[10]", make_datasets_CityScape.file_listX_seg[10])
    print("self.file_listX[10]", make_datasets_CityScape.file_listX[10])


    image_debug_seg1 = (imagesX_seg[0])
    image_debug_seg2 = make_datasets_CityScape.convert_indexInt_to_color(image_debug_seg1)
    image_debug_seg3 = Image.fromarray(image_debug_seg2.astype(np.uint8))
    image_debug_ori = Image.fromarray(((imagesX[0] + 1.0) * 127.5).transpose(1, 2, 0).astype(np.uint8))
    image_concat = make_datasets_CityScape.get_concat_img_h(image_debug_ori, image_debug_seg3)

    image_debug_seg1y = (imagesY_seg[0])
    image_debug_seg2y = make_datasets_CityScape.convert_indexInt_to_color(image_debug_seg1y)
    image_debug_seg3y = Image.fromarray(image_debug_seg2y.astype(np.uint8))
    image_debug_oriy = Image.fromarray(((imagesY[0] + 1.0) * 127.5).transpose(1, 2, 0).astype(np.uint8))
    image_concaty = make_datasets_CityScape.get_concat_img_h(image_debug_oriy, image_debug_seg3y)

    image_big = make_datasets_CityScape.get_concat_img_w(image_concat, image_concaty)



    image_big.show()