import time
from pyqtgraph.Qt import QtGui
import PyQt5
from PyQt5.QtCore import *
import numpy as np
import sys

import drone
import window

init_time = time.time()
def update():
    t = time.time() - init_time
    # print(t)
    drone0.update()
    body_pos = np.array([drone0.pos[0]*1000,drone0.pos[1]*1000,drone0.pos[2]*1000])#mm
    body_ori = np.array([drone0.ori[0]*180/np.pi,drone0.ori[1]*180/np.pi,drone0.ori[2]*180/np.pi])#deg
    prop_pos = np.array([drone0.prop_w[0]*t,-drone0.prop_w[1]*t,drone0.prop_w[2]*t,-drone0.prop_w[3]*t])
    window0.moveMesh(body_pos,body_ori,prop_pos)
    azim_ang = np.arctan2(drone0.pos[1]*1000-window0.init_cam_pos[1],drone0.pos[0]*1000-window0.init_cam_pos[0])*180/np.pi+90
    window0.viewer.setCameraPosition(azimuth=90+azim_ang)

    # print(azim_ang)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window0 = window.window()
    drone0 = drone.drone()
    drone0.window0 = window0
    window0.show()

    t1 = QTimer()
    t1.setInterval(int(drone0.dt*1000))
    t1.timeout.connect(update)
    t1.start()
    sys.exit(app.exec_())


