# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import os
import numpy as np
import cv2
import torch
from config import M, LANDMARK_PATH, PROJECT_DIR
from src.functions import create_shading_recon
from src.mask import MaskGenerator
from src.model import SfSNet
from src.utils import convert

# if __name__ == '__main__':
#     pass


def _decomposition(image_path):
    # define a SfSNet
    net = SfSNet()
    # set to eval mode
    net.eval()
    # load weights
    # net.load_weights_from_pkl('SfSNet-Caffe/weights.pkl')
    net.load_state_dict(torch.load(os.path.join(PROJECT_DIR, 'data/SfSNet.pth')))
    # define a mask generator
    mg = MaskGenerator(LANDMARK_PATH)

    # read image
    image = cv2.imread(image_path)

    # crop face and generate mask of face
    aligned, mask, im, landmark = mg.align(image, size=(M, M))[0]

    # cv2.imshow("im", im)  #
    # cv2.waitKey(0)  #

    # resize
    im = cv2.resize(im, (M, M))
    # normalize to (0, 1.0)
    im = np.float32(im) / 255.0
    # from (128, 128, 3) to (1, 3, 128, 128)
    im = np.transpose(im, [2, 0, 1])
    im = np.expand_dims(im, 0)

    # get the normal, albedo and light parameter
    normal, albedo, light = net(torch.from_numpy(im))

    # get numpy array
    n_out = normal.detach().numpy()
    al_out = albedo.detach().numpy()
    light_out = light.detach().numpy()

    # -----------add by wang-------------
    # from [1, 3, 128, 128] to [128, 128, 3]
    n_out = np.squeeze(n_out, 0)
    n_out = np.transpose(n_out, [1, 2, 0])
    # from [1, 3, 128, 128] to [3, 128, 128]
    al_out = np.squeeze(al_out, 0)
    # from [3, 128, 128] to  to [128, 128, 3]
    al_out = np.transpose(al_out, [1, 2, 0])
    # from [1, 27] to [27, 1]
    light_out = np.transpose(light_out, [1, 0])
    # -----------end---------------------

    # Converting images from BGR format to RGB format
    n_out2 = n_out[:, :, (2, 1, 0)]
    # print 'n_out2 shape', n_out2.shape
    n_out2 = 2 * n_out2 - 1  # [-1 1]
    nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))
    nr = np.expand_dims(nr, axis=2)
    n_out2 = n_out2 / np.repeat(nr, 3, axis=2)
    # print 'nr shape', nr.shape

    # convert to RGB
    al_out2 = al_out[:, :, (2, 1, 0)]

    # Note: n_out2, al_out2, light_out is the actual output
    Irec, Ishd = create_shading_recon(n_out2, al_out2, light_out)

    # diff = (mask // 255)
    # n_out2 = n_out2 * diff
    # al_out2 = al_out2 * diff
    # Ishd = Ishd * diff
    # Irec = Irec * diff

    # -----------add by wang------------
    Ishd = cv2.cvtColor(Ishd, cv2.COLOR_RGB2GRAY)

    # al_out2 = (al_out2 / np.max(al_out2) * 255).astype(dtype=np.uint8)
    # Irec = (Irec / np.max(Irec) * 255).astype(dtype=np.uint8)
    # Ishd = (Ishd / np.max(Ishd) * 255).astype(dtype=np.uint8)

    # convert to BGR
    al_out3 = cv2.cvtColor(al_out2, cv2.COLOR_RGB2BGR)
    n_out3 = cv2.cvtColor(n_out2, cv2.COLOR_RGB2BGR)
    Irec = cv2.cvtColor(Irec, cv2.COLOR_RGB2BGR)
    # -------------end---------------------
    # cv2.imshow("Normal", n_out3)
    # cv2.imshow("Albedo", al_out3)
    # cv2.imshow("Recon", Irec)
    # cv2.imshow("Shading", Ishd)
    # cv2.waitKey(0)
    cv2.imwrite('data/shading.png', convert(Ishd))
    cv2.imwrite('data/Albedo.png', convert(al_out3))

    return [n_out2, al_out2, light_out, al_out3, n_out3, mask]


# if __name__ == '__main__':
#     d_path = os.path.join(PROJECT_DIR, 'data')
#     if not os.path.exists(d_path):
#         os.mkdir(d_path)
#     _decomposition("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/Images/13.png_face.png")
#
