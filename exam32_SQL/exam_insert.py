import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_window = uic.loadUiType('./user_insert.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_insert.clicked.connect(self.btn_insert_slot)

    def btn_insert_slot(self):
        userID = self.le_userid.text()
        userName = self.le_username.text()
        birthYear = self.le_birthyear.text()
        addr = self.le_addr.text()
        mobile = self.le_mobile.text()
        height = self.le_height.text()
        print(userID)
        print(userName)
        print(birthYear)
        print(addr)
        print(mobile)
        print(height)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())