import sys
import os
import cv2
from PyQt5 import QtWidgets, QtGui
from ui import Ui_Dialog
from recognition import predict_expression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class UI(Ui_Dialog):
    def __init__(self, form, model):
        super(UI, self).__init__()
        self.setupUi(form)
        self.model = model
        self.pushButton.clicked.connect(self.openFileDrowser)

    def openFileDrowser(self):
        # 加载模型
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory="./dataset/test/",
                                                                     filter="All Files (*);;Text Files (*.txt)")
        # 显示结果
        if file_name is not None and file_name != "":
            img = predict_expression(file_name, model)
            self.showImg(img)

    def showImg(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img.shape[1]
        y = img.shape[0]
        frame = QtGui.QImage(img, x, y, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(frame)
        item = QtWidgets.QGraphicsPixmapItem(pix)  # 创建像素图元
        scale = 1  # 图片缩放倍数
        if x > 760:
            scale = 760 / x
            if y * scale > 500:
                scale = 500 / y
        item.setScale(scale)
        scene = QtWidgets.QGraphicsScene()  # 创建场景
        scene.addItem(item)
        self.graphicsView.setScene(scene)  # 将场景添加至视图


def load_cnn_model():
    """
    载入CNN模型
    :return:
    """
    from model import CNN
    model = CNN()
    model.load_weights('./models/cnn_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
