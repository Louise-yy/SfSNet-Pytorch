import os
import sys

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtCore
from GUI.albedo import Ui_Form as albedo_ui
from config import PROJECT_DIR
from GUI.error import Ui_Error as error_Gui
from GUI.save import Ui_Save as save_Gui

from SfSNet_test import _decomposition
from alter import albedo_highlight, albedo_bilateral, histogram_matching, unsharp_masking, shading_alter


class albedo_Gui(QWidget, albedo_ui):
    """
    GUI operation of the main page
    """
    img_add = ""
    img2_add = ""
    img_al_out3 = None
    img_n_out2 = None
    img_light_out = None
    img_mask = None
    templete_albedo = None
    templete_light = None

    def __init__(self):
        super(albedo_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)
        self.retranslateUi(self)

    """
    get the original image and decompose it
    """
    def getImage_click(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', '.',
                                                       'Image files (*.jpg *.png)')
        self.img_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic.setScaledContents(True)
        # Show the image on the label
        self.L_ShowPic.setPixmap(image)
        n_out2, al_out2, light_out, al_out3, n_out3, mask = _decomposition(imgName)
        self.img_al_out3 = al_out3
        self.templete_albedo = al_out3
        self.img_n_out2 = n_out2
        self.img_light_out = light_out
        self.templete_light = light_out
        self.img_mask = mask

    """
    get the reference image
    """
    def getImage2_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', '.',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        self.img2_add = imgName
        image = QtGui.QPixmap(imgName).scaled(384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_ShowPic_2.setScaledContents(True)
        self.L_ShowPic_2.setPixmap(image)

    """
    add highlight(button)
    """
    def highlight_click(self):
        if self.img_add == "":  # Determine if there is an input original image
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_highlight(al_out3, n_out2, light_out, mask, 1.25, 1)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/highlight.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after highlight:"))

    """
    add highlight(slider)
    """
    def highlight_slide(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            size = self.sender().value()
            weight = 1 + size / 100
            gamma = size / 10
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_highlight(al_out3, n_out2, light_out, mask, weight, gamma)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/highlight.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after highlight:"))

    """
    add buffing effect(button)
    """
    def buffing_click(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_bilateral(al_out3, n_out2, light_out, mask, 40)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/buffing.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after buffing:"))

    """
    add buffing effect(slider)
    """
    def buffing_slide(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            sigmaColor = self.sender().value()
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = albedo_bilateral(al_out3, n_out2, light_out, mask, sigmaColor)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/buffing.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after buffing:"))

    """
    add sharpening effect(button)
    """
    def sharpening_click(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = unsharp_masking(al_out3, 1, n_out2, light_out, mask)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/sharpening.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after sharpening:"))

    """
    add sharpening effect(slider)
    """
    def sharpening_slide(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        else:
            amount = self.sender().value()
            amount = amount/10
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = unsharp_masking(al_out3, amount, n_out2, light_out, mask)
            self.templete_albedo = albedo
            image = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/sharpening.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after sharpening:"))

    """
    change the albedo by reference
    """
    def reference_click(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        elif self.img2_add == "":
            self.ui = error_Gui()
            self.ui.show()
        else:
            al_out3 = self.img_al_out3
            n_out2 = self.img_n_out2
            light_out = self.img_light_out
            mask = self.img_mask
            albedo = histogram_matching(al_out3, n_out2, light_out, self.img2_add, mask)
            self.templete_albedo = albedo
            image2 = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/matching.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image2)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after matching:"))

    """
    change the lighting by reference
    """
    def lighting_click(self):
        if self.img_add == "":
            self.ui = error_Gui()
            _translate = QtCore.QCoreApplication.translate
            self.ui.L_text1.setText(_translate("Form", "Please upload the original image first"))
            self.ui.show()
        elif self.img2_add == "":
            self.ui = error_Gui()
            self.ui.show()
        else:
            imgName = self.img2_add
            n_out2 = self.img_n_out2
            al_out3 = self.img_al_out3
            mask = self.img_mask
            shading = shading_alter(imgName, n_out2, al_out3, mask)
            self.templete_light = shading
            image2 = QtGui.QPixmap(
                os.path.join(PROJECT_DIR, 'data/f2f.png')).scaled(
                384, 384, aspectRatioMode=Qt.KeepAspectRatio)
            self.L_ShowAfter.setScaledContents(True)
            self.L_ShowAfter.setPixmap(image2)
            _translate = QtCore.QCoreApplication.translate
            self.label.setText(_translate("Form", "The image after relighting:"))

    """
    save the status of the display image
    """
    def save_click(self):
        self.img_al_out3 = self.templete_albedo
        self.img_light_out = self.templete_light
        self.ui = save_Gui()
        self.ui.show()


class error_Gui(QWidget, error_Gui):
    """
    GUI of the error page
    """
    def __init__(self):
        super(error_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)
        image = QtGui.QPixmap(
            os.path.join(PROJECT_DIR, 'data/error.png')).scaled(
            384, 384, aspectRatioMode=Qt.KeepAspectRatio)
        self.L_image.setScaledContents(True)
        self.L_image.setPixmap(image)

    def OK_clicked(self):
        self.close()


class save_Gui(QWidget, save_Gui):
    """
    GUI of the save successfully page
    """
    def __init__(self):
        super(save_Gui, self).__init__()
        self.ui = None
        self.setupUi(self)
        self.retranslateUi(self)

    def close_clicked(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    interface = albedo_Gui()
    interface.show()
    sys.exit(app.exec_())
