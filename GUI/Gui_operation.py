import os
import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from albedo import Ui_Form as albedo_ui
from config import PROJECT_DIR
from home_page import Ui_Form as homePage_ui
from lighting import Ui_Form as lighting_ui
from error import Ui_Form as error_ui

from SfSNet_test import _decomposition
from alter import albedo_highlight, albedo_bilateral, histogram_matching, unsharp_masking, shading_alter


class albedo_Gui(QWidget, albedo_ui):
    img_add = ""
    img2_add = ""
    img_al_out3 = None
    img_n_out2 = None
    img_light_out = None
    img_mask = None

    def __init__(self):
        super(albedo_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)
        self.retranslateUi(self)

    def getImage_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', '.',
                                                       'Image files (*.jpg *.png)')
        self.img_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic.setScaledContents(True)
        # Show the image on the label
        self.L_ShowPic.setPixmap(image)
        n_out2, al_out2, light_out, al_out3, n_out3, mask = _decomposition(imgName)
        self.img_al_out3 = al_out3
        self.img_n_out2 = n_out2
        self.img_light_out = light_out
        self.img_mask = mask

    def getImage2_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', '.',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        self.img2_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic_2.setScaledContents(True)
        self.L_ShowPic_2.setPixmap(image)

    def highlight_click(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_highlight(al_out3, n_out2, light_out, mask, 1.25, 1)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/highlight.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after highlight:"))

    def highlight_slide(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            size = self.sender().value()
            weight = 1 + size / 100
            gamma = size / 10
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_highlight(al_out3, n_out2, light_out, mask, weight, gamma)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/highlight.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after highlight:"))

    def buffing_click(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_bilateral(al_out3, n_out2, light_out, mask, 40)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/buffing.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after buffing:"))

    def buffing_slide(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            sigmaColor = self.sender().value()
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_bilateral(al_out3, n_out2, light_out, mask, sigmaColor)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/buffing.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after buffing:"))

    def sharpening_click(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            unsharp_masking(al_out3, 1, n_out2, light_out, mask)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/sharpening.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after sharpening:"))

    def sharpening_slide(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            amount = self.sender().value()
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            unsharp_masking(al_out3, amount, n_out2, light_out, mask)
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/sharpening.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after sharpening:"))

    def reference_click(self):
        al_out3 = self.img_al_out3
        n_out2 = self.img_n_out2
        light_out = self.img_light_out
        mask = self.img_mask
        histogram_matching(al_out3, n_out2, light_out, self.img2_add, mask)

        image2 = QtGui.QPixmap(
            os.path.join(PROJECT_DIR, 'data/matching.png')).scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowAfter.setScaledContents(True)
        self.L_ShowAfter.setPixmap(image2)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "The image after matching:"))

    def lighting_click(self):
        # print(self.img2_add)
        # if self.img2_add == "":
        #     self.ui = lighting_Gui()
        #     self.ui.show()
        # else:
        imgName = self.img2_add
        n_out2 = self.img_n_out2
        al_out3 = self.img_al_out3
        mask = self.img_mask
        shading_alter(imgName, n_out2, al_out3, mask)

        image2 = QtGui.QPixmap(
            os.path.join(PROJECT_DIR, 'data/f2f.png')).scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowAfter.setScaledContents(True)
        self.L_ShowAfter.setPixmap(image2)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "The image after relighting:"))

    # def returnHP_click(self):
    #     self.ui = HomePage_Gui()
    #     self.ui.show()
    #     self.close()


class error_Gui(QWidget, error_ui):
    def __init__(self):
        super(error_ui, self).__init__()
        self.ui = None
        self.setupUi(self)
        image = QtGui.QPixmap("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/error.png").scaled(384,
                                                                                                                  384,
                                                                                                                  aspectRatioMode=Qt.KeepAspectRatio)
        self.L_image.setScaledContents(True)
        self.L_image.setPixmap(image)
        _translate = QtCore.QCoreApplication.translate
        self.L_text1.setText(_translate("Form", "Please upload the reference image first"))

    def OK_clicked(self):
        self.close()


# class HomePage_Gui(QWidget, homePage_ui):
#     def __init__(self):
#         super(HomePage_Gui, self).__init__()
#         self.ui = None
#         self.setupUi(self)
#
#     def connect_albedo(self):
#         self.ui = albedo_Gui()
#         self.ui.show()
#         self.close()
#
#     def connect_lighting(self):
#         self.ui = lighting_Gui()
#         self.ui.show()
#         self.close()
#
# class lighting_Gui(QWidget, lighting_ui):
#     img_add = ""
#     img_s_add = ""
#     img_al_out3 = None
#     img_n_out2 = None
#     img_light_out = None
#
#     # img_s_light_out = None
#
#     def __init__(self):
#         super(lighting_Gui, self).__init__()
#         self.ui = None
#         self.setupUi(self)
#
#     def getImage_click(self):
#         # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
#         imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
#                                                                           '\final_project\SfSNet-Pytorch',
#                                                        'Image files (*.jpg *.gif *.png *.jpeg)')
#         self.img_add = imgName
#         image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
#         self.L_ShowPic.setScaledContents(True)
#         # Show the image on the label
#         self.L_ShowPic.setPixmap(image)
#         n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(imgName)
#         self.img_al_out3 = al_out3
#         self.img_n_out2 = n_out2
#         self.img_light_out = light_out
#
#     def getImage2_click(self):
#         # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
#         imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
#                                                                           '\final_project\SfSNet-Pytorch',
#                                                        'Image files (*.jpg *.gif *.png *.jpeg)')
#         self.img_s_add = imgName
#         image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
#         self.L_ShowPic_2.setScaledContents(True)
#         self.L_ShowPic_2.setPixmap(image)
#
#         # n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(imgName)
#         # self.img_al_out3 = al_out3
#         # self.img_n_out2 = n_out2
#         # self.img_s_light_out = light_out
#
#     # def lighting_click(self):
#     #     # imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
#     #     #                                                                   '\final_project\SfSNet-Pytorch',
#     #     #                                                'Image files (*.jpg *.gif *.png *.jpeg)')
#     #     # image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
#     #     # self.L_ShowPic_2.setScaledContents(True)
#     #     # self.L_ShowPic_2.setPixmap(image)
#     #
#     #     imgName = self.img_s_add
#     #     al_out3 = self.img_al_out3
#     #     n_out2 = self.img_n_out2
#     #
#     #     shading_alter(imgName, n_out2, al_out3)
#     #
#     #     image2 = QtGui.QPixmap(
#     #         "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/f2f.png").scaled(
#     #         384, 384, aspectRatioMode=Qt.KeepAspectRatio)
#     #     self.L_ShowAfter.setScaledContents(True)
#     #     self.L_ShowAfter.setPixmap(image2)
#     #     _translate = QtCore.QCoreApplication.translate
#     #     self.label.setText(_translate("Form", "The image after relighting:"))
#
#     # def returnHP_click(self):
#     #     self.ui = HomePage_Gui()
#     #     self.ui.show()
#     #     self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # interface = HomePage_Gui()
    interface = albedo_Gui()
    interface.show()
    sys.exit(app.exec_())
