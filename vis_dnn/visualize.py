from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        f, axes = plt.subplots(1,2)
        self.axes = axes
        self.f = f
        
        axes[0].set_facecolor('black')
        axes[1].set_facecolor('black')
        f.set_size_inches((50,100))
        
        self.compute_initial_figure()
        FigureCanvas.__init__(self, f)
        self.setParent(parent)
    def compute_initial_figure(self):
        pass

class ExperimentWidget(QWidget):
    def __init__(self, output, latent,label, epoch, loss):
        QMainWindow.__init__(self)
        
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self)
        vbox.addWidget(self.canvas)
        
        self.setLayout(vbox)
        
        for i in range(len(latent)):
            if label[i].item()==0:
                self.line = self.canvas.axes[0].scatter(x = latent[i][0][0], y = latent[i][0][1], color='red', s= 100)
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'red', s=100)
            elif label[i].item()==1:
                self.line = self.canvas.axes[0].scatter(x = latent[i][0][0], y = latent[i][0][1], color='yellow', s= 100)
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'yellow', s=100)
        
    def update_var(self, output, latent, label, epoch, loss):
        self.canvas.axes[0].cla()
        self.canvas.axes[1].cla()
        self.canvas.f.suptitle(f"Epoch: {epoch} Loss : {loss}", fontsize= 20)
        
        for i in range(len(latent)):
            if label[i].item()==0:
                self.line = self.canvas.axes[0].scatter(x = latent[i][0][0], y = latent[i][0][1], color='red', s= 100)
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'red', s=100)
            elif label[i].item()==1:
                self.line = self.canvas.axes[0].scatter(x = latent[i][0][0], y = latent[i][0][1], color='yellow', s= 100)
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'yellow', s=100)
        plt.pause(0.0001)


class VisualizeAllLayer(QWidget):
    def __init__(self, output, latent,label, epoch, loss):
        QMainWindow.__init__(self)
        
        
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self)
        vbox.addWidget(self.canvas)
        
        self.setLayout(vbox)
        for i in range(len(label)):
            if label[i].item()==0:
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'red', s=100)
            elif label[i].item()==1:
                self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'yellow', s=100)
            for j in range(len(latent[0])):
                if label[i].item()==0:
                    self.line = self.canvas.axes[0].scatter(x = latent[i][j][0][0], y = latent[i][j][0][1], color='red', s= 100)
                elif label[i].item()==1:
                    self.line = self.canvas.axes[0].scatter(x = latent[i][j][0][0], y = latent[i][j][0][1], color='yellow', s= 100)
        plt.pause(3)

    def update_var(self, output, latent, label, epoch, loss):
        self.canvas.axes[0].cla()
        self.canvas.axes[1].cla()

        for j in range(len(latent[0])):
            for i in range(len(label)):
                if label[i].item()==0:
                    self.line = self.canvas.axes[0].scatter(x = latent[i][j][0][0], y = latent[i][j][0][1], color='red', s= 100)
                elif label[i].item()==1:
                    self.line = self.canvas.axes[0].scatter(x = latent[i][j][0][0], y = latent[i][j][0][1], color='yellow', s= 100)
                self.canvas.axes[0].set_title(f"Layer : {j}", fontsize = 20)
                if j == 0:
                    self.canvas.axes[1].set_title(f"Epoch: {epoch} Loss : {loss}", fontsize = 20)
                    if label[i].item()==0:
                        self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'red', s=100)
                    elif label[i].item()==1:
                        self.line2 = self.canvas.axes[1].scatter(output[i], y = 0, color = 'yellow', s=100)
            plt.pause(1)
            self.canvas.axes[0].cla()
        plt.pause(0.001)