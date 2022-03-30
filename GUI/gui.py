import sys


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from Interaction import Ui_Form


from SfSNet_test import _decomposition
from alter import albedo_highlight, albedo_bilateral, albedo_sharp, histogram_matching


class Gui(QWidget, Ui_Form):
    img_add = ""
    img_al_out3 = None
    img_n_out2 = None
    img_light_out = None

    def __init__(self):
        super(Gui, self).__init__()
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
            albedo_sharp(al_out3, n_out2, light_out)
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

        path = self.img_add
        histogram_matching(path, imgName)

        image2 = QtGui.QPixmap(
            "D:/AoriginallyD/Cardiff-year3/final_project/SfSNet-Pytorch/data/matching.png").scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowAfter.setScaledContents(True)
        self.L_ShowAfter.setPixmap(image2)
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "The image after matching:"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    interface = Gui()
    interface.show()
    sys.exit(app.exec_())
