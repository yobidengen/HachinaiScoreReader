#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import pandas as pd
import json
import sqlite3
import os
import datetime

#ローマ字を漢字に変換(適当)
def changeName(name):
    if name == "ichi":
        return "一"
    elif name == "ni":
        return "二"
    elif name == "san":
        return "三"
    elif name == "yuu":
        return "遊"
    elif name == "tou":
        return "投"
    elif name == "hidari":
        return "左"
    elif name == "naka":
        return "中"
    elif name == "migi":
        return "右"
    elif name == "an":
        return "安"
    elif name == "go":
        return "ゴ"
    elif name == "hi":
        return "飛"
    elif name == "hon":
        return "本"
    elif name == "shitsu":
        return "失"
    elif name == "tyoku":
        return "直"
    elif name == "gihi":
        return "犠飛"
    elif name == "shin":
        return "振"
    elif name == "shikyu":
        return "四球"
    elif name == "heisatsu":
        return "併殺"
    else:
        return "unknown"


#OpenCVの画像をPyQtで表示できるように変換
#こちらのソースコードを利用
#http://qiita.com/odaman68000/items/c8c4093c784bff43d319
def create_QPixmap(image):
    qimage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * image.shape[2], QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qimage)
    return pixmap

#テンプレートマッチングの実行
def matching(img,tempImage,threshold,img_res,cell_y,cell_x):
    
    w, h = tempImage.shape[::-1]

    res = cv2.matchTemplate(img,tempImage,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    res_loc = []
    for pt in zip(*loc[::-1]):
        #重複して検出されたものを除外
        flag=True
        for pt2 in res_loc:
            if pt2[0] + w > pt[0]:
                flag = False
        if flag:
            res_loc.append(pt)
            #元の画像の検出部分に枠を描画
            cv2.rectangle(img_res, (pt[0]+cell_x, pt[1]+cell_y), (pt[0]+cell_x+w, pt[1]+cell_y+h), (0,0,255), 2)
    return res_loc

#画像をドロップした際に開くウィンドウ
class Add_widget(QtWidgets.QDialog):

    def __init__(self,orderImage,resultImage,clipboard,parent=None):
        super(Add_widget, self).__init__(parent)
        self.initUI(orderImage,resultImage,clipboard,parent)

    def initUI(self,orderImage,resultImage,clipboard,parent):
        self.lbl = QtWidgets.QLabel()
        self.orderImage = orderImage
        self.resultImage = resultImage

        self.spinlbl = QtWidgets.QLabel("threshold")
        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setRange(0,1)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setValue(0.80)
        self.spinbox.valueChanged.connect(self.get_result)
        self.sbin_hbox = QtWidgets.QHBoxLayout()
        self.sbin_hbox.addWidget(self.spinlbl)
        self.sbin_hbox.addWidget(self.spinbox)
        self.sbin_hbox.addStretch(1)

        self.button = QtWidgets.QPushButton("CSVファイルの書き込み")
        self.button.clicked.connect(self.writeCsv)

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.lbl)
        #self.vbox.addWidget(self.datatable)
        self.vbox.addLayout(self.sbin_hbox)
        self.vbox.addWidget(self.button)
        self.setLayout(self.vbox)
        self.setWindowTitle('result')
        self.clipboard = clipboard

        self.get_result()

    #csv書き込み
    def writeCsv(self):
        now_ = datetime.datetime.today()
        self.df.to_csv('score{}.csv'.format(now_.strftime("%Y-%m-%d-%H-%M-%S")))
        self.accept()

    #文字の検出
    def detection_value(self,frame,threshold):

        try:
            f = open("player.json", 'r')
            player_data = json.load(f)
        except UnicodeDecodeError:
            f = open("player.json", 'r', encoding='utf-8')
            player_data = json.load(f)

        img_res = frame.copy()
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        
        df = pd.DataFrame()
        
        #1列ごとに文字を検出
        for col in range(int((self.resultImage.shape[1]+30)/180)):

            #イニング検出
            tempList = os.listdir("template/inning")
            inningList = []

            for row in range(12):
                cell_y = 287-280 + (34+29)*row
                cell_x = 631-631 + 180*col
                img_cell = img_gray[cell_y:cell_y+29,cell_x:cell_x+39]
                list_num = []

                #イニング画像でテンプレートマッチングを行う
                for template in tempList:
                    tempImage = cv2.imread(os.path.join('./template/inning',template),0)
                    loc = matching(img_cell,tempImage,0.95,img_res,cell_y,cell_x)
                    for pt in loc:
                        list_num.append([template[:-4],pt[0],pt[1]])

                #x座標でソートする
                list_num.sort(key=lambda x:(x[1]))
                if len(list_num) > 0:
                    res = "".join(np.array(list_num)[:,0])
                else:
                    res = ""

                inningList.append(res)

            df[col*3] = pd.Series(data=inningList)

            #打席結果検出
            tempList = os.listdir("template/result")
            resultList = []

            for row in range(12):
                cell_y = 301-280 + (34+29)*row
                cell_x = 678-631 + 180*col
                img_cell = img_gray[cell_y:cell_y+34,cell_x:cell_x+82]
                list_num = []

                #打席結果画像でテンプレートマッチングを行う
                for template in tempList:
                    tempImage = cv2.imread(os.path.join('./template/result',template),0)
                    loc = matching(img_cell,tempImage,threshold,img_res,cell_y,cell_x)
                    for pt in loc:
                        list_num.append([changeName(template[:-4]),pt[0],pt[1]])

                #x座標でソートする
                list_num.sort(key=lambda x:(x[1]))
                if len(list_num) > 0:
                    res = "".join(np.array(list_num)[:,0])
                else:
                    res = ""

                resultList.append(res)
            df[col*3+1] = pd.Series(data=resultList)

            #打点検出
            tempList = os.listdir("template/RBI")
            RBIList = []
            for row in range(12):
                cell_y = 282-280 + (34+29)*row
                cell_x = 741-631 + 180*col
                img_cell = img_gray[cell_y:cell_y+35,cell_x:cell_x+36]
                list_num = []

                #打点画像でテンプレートマッチングを行う
                for template in tempList:
                    tempImage = cv2.imread(os.path.join('./template/RBI',template),0)
                    loc = matching(img_cell,tempImage,0.95,img_res,cell_y,cell_x)
                    for pt in loc:
                        list_num.append([template[:-4],pt[0],pt[1]])

                #x座標でソートする
                list_num.sort(key=lambda x:(x[1]))
                if len(list_num) > 0:
                    res = "".join(np.array(list_num)[:,0])
                else:
                    res = ""

                RBIList.append(res)
            df[col*3+2] = pd.Series(data=RBIList)

        self.df = df
        return img_res

    #成績を取得する
    def get_result(self):
        img_res = self.detection_value(self.resultImage,self.spinbox.value())
        mergeImage = np.concatenate([self.orderImage,img_res],axis=1)
        mergeImage = cv2.cvtColor(mergeImage, cv2.COLOR_BGR2RGB)
        mergeImage = cv2.resize(mergeImage, (1280,720))
        qt_img = create_QPixmap(mergeImage)
        self.lbl.setPixmap(qt_img)

    def show(self):
        self.exec_()

#ドラッグアンドドロップに対応したQLabelクラス
class DropLabel(QtWidgets.QLabel):
    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setAlignment(QtCore.Qt.AlignCenter);
        self.setText("Drop here.")

    def dragEnterEvent(self, e):
            e.accept()

    # 2つの画像を合成(適当)
    def mergeResultImage(self,img1,img2):
        minMad = 1000
        minIndex = 0
        min12 = 0
        for i in range(0,400,2):
            mad = np.mean(np.abs(img1[:400,i:,:] - img2[:400,:img2.shape[1]-i,:]))

            if mad < minMad:
                minMad = mad
                minIndex = i
                min12 = 0

        for i in range(0,400,2):
            mad = np.mean(np.abs(img2[:400,i:,:] - img1[:400,:img1.shape[1]-i,:]))

            if mad < minMad:
                minMad = mad
                minIndex = i
                min12 = 1

        if minIndex != 0:
            if min12 == 0:
                return np.concatenate([img1,img2[:,img2.shape[1]-minIndex:,:]],axis=1)
            else:
                return np.concatenate([img2,img1[:,img1.shape[1]-minIndex:,:]],axis=1)
        else:
            return img1

    def dropEvent(self, e):
        mimeData = e.mimeData()

        files = [u.toLocalFile() for u in mimeData.urls()]
        print("loading {}".format(files[0]))
        #ドロップされた画像を読み込み
        frame = cv2.imread(files[0])
        #読み込みに失敗した場合は処理を行わない
        if frame is not None:
            frame = cv2.resize(frame, self.parent.size)
            orderImage = frame[280:1043,356:614,:]
            #画像が1枚の時
            if len(files) == 1:
                resultTmage = frame[280:1043,631:1711,:]
            #画像が2枚の時
            else:
                resultTmage = frame[280:1043,631:1649,:]
                frame2 = cv2.imread(files[1])
                if frame2 is not None:
                    resultTmage2 = frame2[280:1043,631:1649,:]
                    resultTmage = self.mergeResultImage(resultTmage,resultTmage2)

            add_widget = Add_widget(orderImage,resultTmage,self.parent.clipboard,self)
            add_widget.show()

#画像をドロップするウィンドウ
class Hachinai_widget(QtWidgets.QWidget):

    def __init__(self,clipboard=None,parent=None):
        super(Hachinai_widget, self).__init__(parent)
        super().__init__()

        self.initUI(clipboard,parent)

    def initUI(self,clipboard,parent):
        self.parent=parent
        self.height = 1080
        self.width = 1920
        self.size = (self.width,self.height)
        self.clipboard = clipboard

        self.lbl = DropLabel(self)
        self.lbl.setMinimumSize(640,480)
        self.lbl.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)

        self.vbox = QtWidgets.QVBoxLayout()
        self.vbox.addWidget(self.lbl)
        self.setLayout(self.vbox)
        self.setWindowTitle('hachinai')
        self.show()
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    clipboard = app.clipboard()
    screen = Hachinai_widget(clipboard)
    sys.exit(app.exec_())