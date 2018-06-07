import numpy as np
import os
from PIL import Image


def convert_to_10class(d):
    d_mod = np.zeros((len(d), 10), dtype=np.float32)
    for num, contents in enumerate(d):
        d_mod[num][int(contents)] = 1.0
    # debug
    print("d_mod[100] =", d_mod[100])
    print("d_mod[200] =", d_mod[200])

    return d_mod

def make_1_img(img_batch):  # for debug
    for num, ele in enumerate(img_batch):
        if num != 0:
            continue
        print("ele.shape =", ele.shape)
        img_tmp = ele.reshape(28, 28, 1)
        img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
        img_tmp = img_tmp.astype(np.uint8)
        image_PIL = Image.fromarray(img_tmp)
        try:
            os.mkdir('./out_images_tripleGAN')
        except:
            pass
        image_PIL.save("./out_images_tripleGAN/debug_img_" + ".png")

    return
    
def make_output_img(img_X, img_X2Y, img_X2Y2X, img_Y, img_Y2X, img_Y2X2Y, out_image_dir, epoch):
    # print('img_X.shape', img_X.shape)
    # print('img_X2Y.shape', img_X2Y.shape)
    # print('img_X2Y2X.shape', img_X2Y2X.shape)
    # print('img_Y.shape', img_Y.shape)
    # print('img_Y2X.shape', img_Y2X.shape)
    # print('img_Y2X2Y.shape', img_Y2X2Y.shape)

    # print("type(image_array)", type(image_array))
    # print("image_array.shape =", image_array.shape)
    # print("np.max(image_array) = ", np.max(image_array))
    # print("np.min(image_array) = ", np.min(image_array))
    # print("np.mean(image_array) = ", np.mean(image_array))
    wide_image = np.zeros((img_X.shape[0] * img_X.shape[1], img_X.shape[2] * 6, img_X.shape[3]), dtype=np.float32)
    for num in range(img_X.shape[0]):
        # img_rows.append(np.concatenate(
        #     (img_X[num], img_X2Y[num], img_X2Y2X[num], img_Y[num], img_Y2X[num], img_Y2X2Y[num]), axis=2))
        img_row = np.concatenate(
            # (img_X[num], img_X2Y[num], img_X2Y2X[num], img_Y[num], img_Y2X[num], img_Y2X2Y[num]), axis=1)
            (img_X[num], img_X2Y[num], img_X2Y2X[num], img_Y[num], img_Y2X[num], img_Y2X2Y[num]), axis=1)

        wide_image[num * img_X.shape[1]:(num + 1) * img_X.shape[1], :, :] = img_row

    # for num in range(img_X.shape[0]):
    #     for h in range(sample_num_h):
    #         for h_mnist in range(28):
    #             for w_mnist in range(28):
    #                 value_ = image_array[h * sample_num_h + w][h_mnist][w_mnist][0]
    #                 if value_ < 0:
    #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 0.0
    #                     # print("under 0")
    #                 elif value_ > 1:
    #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = 1.0
    #                     # print("over 1")
    #                 else:
    #                     wide_image[h * 28 + h_mnist][w * 28 + w_mnist][0] = value_
    # wide_image = wide_image * 255
    wide_image = (wide_image + 1) * 127.5
    # wide_image = np.tile(wide_image, (1, 1, 3)) * 255
    wide_image = wide_image.astype(np.uint8)
    wide_image_PIL = Image.fromarray(wide_image)
    wide_image_PIL.save(out_image_dir + "/resultImage2_" + str(epoch) + ".png")

    
    return




