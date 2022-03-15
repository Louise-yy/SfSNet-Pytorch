import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from Interaction import Ui_Form
from PIL import Image
from alter import change_albedo


class Gui(QWidget, Ui_Form):
    def __init__(self):
        super(Gui, self).__init__()
        self.setupUi(self)

    def getImage_click(self):
        # Open the file(*.jpg *.gif *.png *.jpeg) from D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images
        imgName, imgType = QFileDialog.getOpenFileName(self, 'Open file', 'D:\AoriginallyD\Cardiff-year3\final_project\SfSNet-Pytorch\Images',
                                                       'Image files (*.jpg *.gif *.png *.jpeg)')
        image = QtGui.QPixmap(imgName)
        # Show the image on the label
        self.L_ShowPic.setPixmap(image)

    def sharpening_click(self):
        image = QtGui.QPixmap(change_albedo())
        self.L_ShowAfter.setPixmap(image)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    interface = Gui()
    interface.show()
    sys.exit(app.exec_())
