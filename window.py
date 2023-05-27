from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  
import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation

import cv2

class window(QMainWindow):
    mesh0 = None
    mesh1 = None
    mesh2 = None
    mesh3 = None
    mesh4 = None

    ground = None
    scttrPlt = None
    scttrHis = None

    textHeight = None

    scatter_cycle = 0
    scatter_period = 10
    scatter_data = []

    region_size = 18000
    region_space = 1000

    init_cam_pos = None
    def __init__(self):
        super(window, self).__init__()
        self.setGeometry(0, 0, 1440, 1080) 
        self.setAcceptDrops(True)
        self.setWindowTitle("drone sim")

        self.initUI()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)

        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        self.viewer.setBackgroundColor("k")
        layout.addWidget(self.viewer, 1)

        tmp = self.viewer.cameraPosition()
        tmp[0] = 0
        tmp[1] = 7000
        tmp[2] = 2000
        self.viewer.setCameraPosition(pos=tmp,distance=1000,elevation=10,azimuth=90)
        self.init_cam_pos = self.viewer.cameraPosition()

        g = gl.GLGridItem()
        g.setSize(self.region_size, self.region_size)
        g.setSpacing(self.region_space, self.region_space)
        self.viewer.addItem(g)

        self.genMesh()

        body_pos = np.array([0,0,20])#mm
        body_ori = np.array([0,0,0])#deg
        prop_pos = np.array([0,0,0,0])
        self.moveMesh(body_pos,body_ori,prop_pos)

        self.scatter_cycle = 0

    def genMesh(self):
        points0, faces0 = self.loadSTL('drone_bodyL.stl')
        points1, faces1 = self.loadSTL('propL.stl')
        points2, faces2 = self.loadSTL('prop_ccwL.stl')
        points3, faces3 = self.loadSTL('propL.stl')
        points4, faces4 = self.loadSTL('prop_ccwL.stl')

        points5, faces5 = self.loadSTL('fail_area.stl')
        points6, faces6 = self.loadSTL('deduction_area.stl')
        points7, faces7 = self.loadSTL('trajectory_line.stl')

        meshdata0 = gl.MeshData(vertexes=points0, faces=faces0)
        meshdata1 = gl.MeshData(vertexes=points1, faces=faces1)
        meshdata2 = gl.MeshData(vertexes=points2, faces=faces2)
        meshdata3 = gl.MeshData(vertexes=points3, faces=faces3)
        meshdata4 = gl.MeshData(vertexes=points4, faces=faces4)

        meshdata5 = gl.MeshData(vertexes=points5, faces=faces5)
        meshdata6 = gl.MeshData(vertexes=points6, faces=faces6)
        meshdata7 = gl.MeshData(vertexes=points7, faces=faces7)
        
        self.mesh0 = gl.GLMeshItem(meshdata=meshdata0,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 1, 1, 1))
        self.mesh1 = gl.GLMeshItem(meshdata=meshdata1,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 0, 0, 1))
        self.mesh2 = gl.GLMeshItem(meshdata=meshdata2,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 0, 0, 1))
        self.mesh3 = gl.GLMeshItem(meshdata=meshdata3,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 0, 1, 1))
        self.mesh4 = gl.GLMeshItem(meshdata=meshdata4,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 0, 1, 1))

        mesh5 = gl.GLMeshItem(meshdata=meshdata5,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 0, 0, 1))
        mesh6 = gl.GLMeshItem(meshdata=meshdata6,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 1, 0, 1))
        mesh7 = gl.GLMeshItem(meshdata=meshdata7,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 1, 0, 1))

        tmp = np.linspace(-self.region_size/2, self.region_size/2, int(self.region_size/2/1000))
        tmp_grid,empty  = np.meshgrid(tmp, tmp)
        z = np.zeros([tmp_grid.shape[0],tmp_grid.shape[1]])
        # self.ground = gl.GLSurfacePlotItem(x=tmp, y=tmp, z=z, color=(0.0, 0.0, 0.0, 1))

        self.scttrPlt = gl.GLScatterPlotItem(pos=np.array([0,0,0]), size=20, color=(1.0, 0.0, 0.0, 1), pxMode=False)
        self.scttrHis = gl.GLScatterPlotItem(pos=np.array([0,0,0]), size=20, color=(1.0, 0.0, 0.0, 1), pxMode=False)

        self.textHeight = gl.GLTextItem(pos=np.array([0,0,0]),text="")

        # self.viewer.addItem(self.ground)

        self.viewer.addItem(mesh5)
        self.viewer.addItem(mesh6)
        self.viewer.addItem(mesh7)

        self.viewer.addItem(self.scttrPlt)
        self.viewer.addItem(self.scttrHis)
        self.viewer.addItem(self.mesh0)
        self.viewer.addItem(self.mesh1)
        self.viewer.addItem(self.mesh2)
        self.viewer.addItem(self.mesh3)
        self.viewer.addItem(self.mesh4)

        self.viewer.addItem(self.textHeight)



    def loadSTL(self, filename):
        m = mesh.Mesh.from_file(filename)
        shape = m.points.shape
        points = m.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        return points, faces
    
    def moveMesh(self,body_pos,body_ori,prop_pos):
        
        if self.scatter_cycle > self.scatter_period:
            self.scatter_data.append([body_pos[0],body_pos[1],0.0])
            self.scttrHis.setData(pos=np.array(self.scatter_data))
            self.scatter_cycle = 0
        else:
            self.scatter_cycle = self.scatter_cycle + 1

        self.scttrPlt.setData(pos=[body_pos[0],body_pos[1],0.0])
        self.textHeight.setData(pos=[body_pos[0],body_pos[1],body_pos[2]+100],text='{:.3g}[m]'.format(body_pos[2]/1000))

        rot = Rotation.from_euler('XYZ', -body_ori/180*np.pi)
        body_pos = rot.apply(body_pos)
        self.mesh0.resetTransform()
        self.mesh1.resetTransform()
        self.mesh2.resetTransform()
        self.mesh3.resetTransform()
        self.mesh4.resetTransform()

        self.mesh0.rotate(body_ori[2],0,0,1,True)
        self.mesh0.rotate(body_ori[1],0,1,0,True)
        self.mesh0.rotate(body_ori[0],1,0,0,True)
        
        self.mesh1.rotate(body_ori[2],0,0,1,True)
        self.mesh1.rotate(body_ori[1],0,1,0,True)
        self.mesh1.rotate(body_ori[0],1,0,0,True)
        
        self.mesh2.rotate(body_ori[2],0,0,1,True)
        self.mesh2.rotate(body_ori[1],0,1,0,True)
        self.mesh2.rotate(body_ori[0],1,0,0,True)
        
        self.mesh3.rotate(body_ori[2],0,0,1,True)
        self.mesh3.rotate(body_ori[1],0,1,0,True)
        self.mesh3.rotate(body_ori[0],1,0,0,True)
        
        self.mesh4.rotate(body_ori[2],0,0,1,True)
        self.mesh4.rotate(body_ori[1],0,1,0,True)
        self.mesh4.rotate(body_ori[0],1,0,0,True)

        self.mesh0.translate( body_pos[0], body_pos[1], body_pos[2],True)
        self.mesh1.translate( body_pos[0] + 35.3553*2.5, body_pos[1] + 35.3553*2.5, body_pos[2] + 15*2.5,True)
        self.mesh2.translate( body_pos[0] + 35.3553*2.5, body_pos[1] - 35.3553*2.5, body_pos[2] + 15*2.5,True)
        self.mesh3.translate( body_pos[0] - 35.3553*2.5, body_pos[1] - 35.3553*2.5, body_pos[2] + 15*2.5,True)
        self.mesh4.translate( body_pos[0] - 35.3553*2.5, body_pos[1] + 35.3553*2.5, body_pos[2] + 15*2.5,True)

        self.mesh1.rotate(prop_pos[0],0,0,1,True)
        self.mesh2.rotate(prop_pos[1],0,0,1,True)
        self.mesh3.rotate(prop_pos[2],0,0,1,True)
        self.mesh4.rotate(prop_pos[3],0,0,1,True)

        

        

if __name__ == '__main__':
    print("window module")