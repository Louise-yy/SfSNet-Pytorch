import sys


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from albedo import Ui_Form as albedo_ui
from home_page import Ui_Form as homePage_ui
from lighting import Ui_Form as lighting_ui

from SfSNet_test import _decomposition
from alter import albedo_highlight, albedo_bilateral, histogram_matching, unsharp_masking, shading_alter


class albedo_Gui(QWidget, albedo_ui):
    img_add = ""
    img_al_out3 = None
    img_n_out2 = None
    img_light_out = None

    def __init__(self):
        super(albedo_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)
        self.retranslateUi(self)

    def getImage_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
                                                                          '\final_project\SfSNet-Pytorch',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        self.img_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic.setScaledContents(True)
        # Show the image on the label
        self.L_ShowPic.setPixmap(image)
        n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(imgName)
        self.img_al_out3 = al_out3
        self.img_n_out2 = n_out2
        self.img_light_out = light_out

    def highlight_click(self):
        if self.img_add == " ":
            print("you haven't choose a picture yet")
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            albedo_highlight(al_out3, n_out2, light_out, 1.25, 1)
            image = QtGui.QPixmap("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/highlight.png").scaled(
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
            # print(size)
            weight = 1 + size/100
            # print(weight)
            gamma = size/10
            # print(gamma)
            img_path = self.img_add
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            albedo_highlight(al_out3, n_out2, light_out, weight, gamma)
            image = QtGui.QPixmap("D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/highlight.png").scaled(
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
            albedo_bilateral(al_out3, n_out2, light_out, 40)
            image = QtGui.QPixmap(
                "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/buffing.png").scaled(
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
            albedo_bilateral(al_out3, n_out2, light_out, sigmaColor)
            image = QtGui.QPixmap(
                "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/buffing.png").scaled(
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
            unsharp_masking(al_out3, 1, n_out2, light_out)
            image = QtGui.QPixmap(
                "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/sharpening.png").scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after sharpening:"))

    def reference_click(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
                                                                          '\final_project\SfSNet-Pytorch',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic_2.setScaledContents(True)
        # Show the image on the label
        self.L_ShowPic_2.setPixmap(image)

        al_out3 = self.img_al_out3
        n_out2 = self.img_n_out2
        light_out = self.img_light_out
        histogram_matching(al_out3, n_out2, light_out, imgName)

        image2 = QtGui.QPixmap(
            "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/matching.png").scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowAfter.setScaledContents(True)
        self.L_ShowAfter.setPixmap(image2)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "The image after matching:"))
        
    def returnHP_click(self):
        self.ui = HomePage_Gui()
        self.ui.show()
        self.close()


class HomePage_Gui(QWidget, homePage_ui):
    def __init__(self):
        super(HomePage_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)

    def connect_albedo(self):
        self.ui = albedo_Gui()
        self.ui.show()
        self.close()

    def connect_lighting(self):
        self.ui = lighting_Gui()
        self.ui.show()
        self.close()


class lighting_Gui(QWidget, lighting_ui):
    img_add = ""
    img_s_add = ""
    img_al_out3 = None
    img_n_out2 = None
    img_light_out = None
    # img_s_light_out = None

    def __init__(self):
        super(lighting_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)

    def getImage_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
                                                                          '\final_project\SfSNet-Pytorch',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        self.img_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic.setScaledContents(True)
        # Show the image on the label
        self.L_ShowPic.setPixmap(image)
        n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(imgName)
        self.img_al_out3 = al_out3
        self.img_n_out2 = n_out2
        self.img_light_out = light_out

    def getImage2_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
                                                                          '\final_project\SfSNet-Pytorch',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        self.img_s_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic_2.setScaledContents(True)
        self.L_ShowPic_2.setPixmap(image)

        # n_out2, al_out2, light_out, al_out3, n_out3 = _decomposition(imgName)
        # self.img_al_out3 = al_out3
        # self.img_n_out2 = n_out2
        # self.img_s_light_out = light_out

    def lighting_click(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3'
        #                                                                   '\final_project\SfSNet-Pytorch',
        #                                                'Image files (*.jpg *.gif *.png *.jpeg)')
        # image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        # self.L_ShowPic_2.setScaledContents(True)
        # self.L_ShowPic_2.setPixmap(image)

        imgName =self.img_s_add
        al_out3 = self.img_al_out3
        n_out2 = self.img_n_out2

        shading_alter(imgName, n_out2, al_out3)

        image2 = QtGui.QPixmap(
            "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/f2f.png").scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowAfter.setScaledContents(True)
        self.L_ShowAfter.setPixmap(image2)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "The image after relighting:"))

    def returnHP_click(self):
        self.ui = HomePage_Gui()
        self.ui.show()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    interface = HomePage_Gui()
    # interface = lighting_Gui()
    interface.show()
    sys.exit(app.exec_())