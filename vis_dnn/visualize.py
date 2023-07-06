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

class AnimationWidget(QWidget):
    def __init__(self, output, latent,label, epoch, loss):
        QMainWindow.__init__(self)
        
        self.latent = latent
        self.output = output
        self.label = label
        self.epoch = epoch
        self.loss = loss
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self)
        vbox.addWidget(self.canvas)
        self.setWindowTitle(f"Epoch: {epoch} Loss : {loss}")
        
        self.setLayout(vbox)
        for i in range(len(latent)):
            if self.label[i].item()==0:
                self.line = self.canvas.axes[0].scatter(x = self.latent[i][0][0], y = self.latent[i][0][1], color='red', s= 100)
                self.line2 = self.canvas.axes[1].scatter(self.output[i], y = 0, color = 'red', s=100)
            elif self.label[i].item()==1:
                self.line = self.canvas.axes[0].scatter(x = self.latent[i][0][0], y = self.latent[i][0][1], color='yellow', s= 100)
                self.line2 = self.canvas.axes[1].scatter(self.output[i], y = 0, color = 'yellow', s=100)
        
    def update_var(self, output, latent, label, epoch, loss):
        self.canvas.axes[0].cla()
        self.canvas.axes[1].cla()
        self.output = output
        self.latent = latent
        self.label = label
        self.setWindowTitle(f"Epoch: {epoch} Loss : {loss}")
        for i in range(len(latent)):
            if self.label[i].item()==0:
                self.line = self.canvas.axes[0].scatter(x = self.latent[i][0][0], y = self.latent[i][0][1], color='red', s= 100)
                self.line2 = self.canvas.axes[1].scatter(self.output[i], y = 0, color = 'red', s=100)
            elif self.label[i].item()==1:
                self.line = self.canvas.axes[0].scatter(x = self.latent[i][0][0], y = self.latent[i][0][1], color='yellow', s= 100)
                self.line2 = self.canvas.axes[1].scatter(self.output[i], y = 0, color = 'yellow', s=100)
        plt.pause(0.001)