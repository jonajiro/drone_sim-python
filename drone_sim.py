import time
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  

import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation

import sys

import pygame
from pygame.locals import *

init_time = time.time()
window = None
drone0 = None

class drone():
    dt = 0.01
    grav = 9.81#重力加速度[m/s2]
    mass = 0.1#質量[kg]
    arm_length = 0.05#ドローンの重心からモーターまでの長さ[m]
    body_width = arm_length*2/np.sqrt(2)#ドローンの幅[m]
    prop_offset = 0.013#モーターからプロペラまでの高さ[m]

    roll_T = 0.6025#２点吊り法の振れ周期[s]
    pitch_T = 0.6025#２点吊り法の振れ周期[s]
    yaw_T = 0.676666667#２点吊り法の振れ周期[s]
    Ixx = mass*grav*(body_width/2)**2/(4*0.16066*(2*np.pi/roll_T)**2)#x軸慣性モーメント[kgm2]
    Iyy = mass*grav*(body_width/2)**2/(4*0.16066*(2*np.pi/pitch_T)**2)#y軸慣性モーメント[kgm2]
    Izz = mass*grav*(arm_length*2/2)**2/(4*0.16066*(2*np.pi/yaw_T)**2)#z軸慣性モーメント[kgm2]
    I = np.array([[Ixx,0,0],[0,Iyy,0],[0,0,Izz]],dtype=float)#慣性モーメント行列[kgm2]

    mf_coef = 0.052439*grav#推力係数[N/duty]
    mm_coef = 0.00063#反トルク係数[Nm/duty]

    pos = np.array([0,0,0],dtype=float)#重心位置[m]
    vel = np.array([0,0,0],dtype=float)#重心速度[m/s]
    omg = np.array([0,0,0],dtype=float)#重心角速度[rad/s]
    ori = np.array([0,0,0],dtype=float)#重心姿勢(ZYXオイラー)[rad]
    ori_q = np.array([0,0,0,0],dtype=float)#重心姿勢(クオータニオン)[q1,q2,q3,w]
    ori_c = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=float)#重心姿勢(回転行列)[3x3]

    ref_vel = np.array([0,0,0],dtype=float)#目標速度[m/s]
    ref_ori = np.array([0,0,0],dtype=float)#目標重心姿勢(ZYXオイラー)[rad]
    ref_height = 0.02#目標高さ[m]
    motor_duty = np.array([0.0,0.0,0.0,0.0],dtype=float)#モーター指示値(0~1)[-]
    com_duty = np.array([0.0,0.0,0.0,0.0],dtype=float)#rpyt指示値(0~1)[-]

    prop_w = np.array([0,0,0,0],dtype=float)#プロペラ回転速度[rad/s]

    state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float)#状態変数

    joy = None
    def __init__(self):
        pygame.joystick.init()
        try:
            self.joy = pygame.joystick.Joystick(0)
        except pygame.error:
            print("Joystick is not found")
        pygame.init()

        self.pos = np.array([0,0,0.02],dtype=float)#m
        self.vel = np.array([0,0,0],dtype=float)
        self.ori = np.array([5*np.pi/180,0,0],dtype=float)#重心姿勢(ZYXオイラー)[rad]
        self.omg = np.array([0,0,0],dtype=float)#重心角速度[rad/s]
        self.ori_c = self.e2c(self.ori)
        self.ori_q = self.c2q(self.ori_c)
        self.set_state()

    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float)#状態変数
        self.pos = np.array([0,0,0.02],dtype=float)#m
        self.vel = np.array([0,0,0],dtype=float)
        self.ori = np.array([0,0,0],dtype=float)#重心姿勢(ZYXオイラー)[rad]
        self.omg = np.array([0,0,0],dtype=float)#重心角速度[rad/s]
        self.ori_c = self.e2c(self.ori)
        self.ori_q = self.c2q(self.ori_c)
        self.ref_height = 0.02#目標高さ[m]
        self.set_state()

    def set_state(self):
        self.state = np.array([self.omg[0],self.omg[1],self.omg[2],self.ori_q[0],self.ori_q[1],self.ori_q[2],self.ori_q[3],self.vel[0],self.vel[1],self.vel[2],self.pos[0],self.pos[1],self.pos[2]],dtype=float)#状態変数
    
    def get_state(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()


        self.ref_vel[0] = float((0.3-(-0.3))*(-self.joy.get_axis(3) + 0.0)/1.0)#r
        self.ref_vel[1] = float((0.3-(-0.3))*(-self.joy.get_axis(2) + 0.0)/1.0)#p
        self.ref_vel[2] = float((1.0-(-1.0))*(-self.joy.get_axis(1) + 0.0)/1.0)#t
        # self.ref_ori[0] = float((10.0*np.pi/180-(-10.0*np.pi/180))*(self.joy.get_axis(2) + 0.0)/1.0)#r
        # self.ref_ori[1] = float((10.0*np.pi/180-(-10.0*np.pi/180))*(-self.joy.get_axis(3) + 0.0)/1.0)#p
        self.ref_ori[2] = float((1.0-(-1.0))*(-self.joy.get_axis(0) + 0.0)/1.0)#y

        # self.ref_height = float((1.0-(-1.0))*(-self.joy.get_axis(1) + 0.0)/1.0)#t
        self.pos = np.array([self.state[10],self.state[11],self.state[12]],dtype=float)
        self.vel = np.array([self.state[7],self.state[8],self.state[9]],dtype=float)
        self.ori_q = np.array([self.state[3],self.state[4],self.state[5],self.state[6]],dtype=float)
        self.omg = np.array([self.state[0],self.state[1],self.state[2]],dtype=float)
        self.ori_c = self.q2c(self.ori_q)
        self.ori = self.c2e(self.ori_c)

        #max 3000RPM 50RPS 2*pi*50
        self.prop_w[0] = self.motor_duty[0]*2*np.pi*1000/60
        self.prop_w[1] = self.motor_duty[1]*2*np.pi*1000/60
        self.prop_w[2] = self.motor_duty[2]*2*np.pi*1000/60
        self.prop_w[3] = self.motor_duty[3]*2*np.pi*1000/60


    def update(self):
        if self.joy.get_button(9) == 1:
            self.reset()
            window.scatter_data = []
            window.scttrPlt.setData(pos=np.array([0,0,0]))
            window.scttrHis.setData(pos=np.array([0,0,0]))

        self.state = self.runge_kutta(self.diff_eq,self.state,self.dt)
        self.get_state()

        # q4t = self.c2q(self.e2c(self.ref_ori))
        # alp = -1.0
        # beta = -0.05
        # q4a = np.array([-self.ori_q[0],-self.ori_q[1],-self.ori_q[2],self.ori_q[3]],dtype=float)
        # q4err = self.qdot(q4t,q4a)
        # qerr = np.array([q4err[0],q4err[1],q4err[2]],dtype=float)
        # u = alp*qerr + beta*(-self.omg)
        # self.com_duty[0] = u[0]
        # self.com_duty[1] = u[1]
        # self.com_duty[2] = u[2]


        xp = -0.6
        yp = -xp
        zp = 0.01

        self.ref_ori[0] = xp * (self.ref_vel[1] - self.vel[1])
        self.ref_ori[1] = yp * (self.ref_vel[0] - self.vel[0])
        self.ref_height = self.ref_height + zp * self.ref_vel[2]
        if self.ref_height < 0.02:
            self.ref_height = 0.02

        rp = -0.9
        rd = -0.05
        pp = rp
        pd = rd
        yp = -0.0
        yd = -0.5

        self.com_duty[0] = rp * (self.ref_ori[0] - self.ori[0]) + rd * (0 - self.omg[0])
        self.com_duty[1] = pp * (self.ref_ori[1] - self.ori[1]) + pd * (0 - self.omg[1])
        self.com_duty[2] = yp * (self.ref_ori[2] - self.ori[2]) + yd * (self.ref_ori[2] - self.omg[2])

        hp = 150.0
        hd = 60.0
        hi = -2.0
        h_trim = self.mass*self.grav / self.mf_coef *1.0
        
        self.com_duty[3] = hp * (self.ref_height - self.pos[2]) + hd * (0 - self.vel[2]) + hi * (0 - self.com_duty[3]) + h_trim
        self.com_duty[3] = self.com_duty[3] / 4

        self.motor_duty[0] = -self.com_duty[0] + self.com_duty[1] + self.com_duty[2] + self.com_duty[3]
        self.motor_duty[1] = +self.com_duty[0] + self.com_duty[1] - self.com_duty[2] + self.com_duty[3]
        self.motor_duty[2] = +self.com_duty[0] - self.com_duty[1] + self.com_duty[2] + self.com_duty[3]
        self.motor_duty[3] = -self.com_duty[0] - self.com_duty[1] - self.com_duty[2] + self.com_duty[3]

        for i in range(4):
            if self.motor_duty[i] > 1:
                self.motor_duty[i] = 1
            elif self.motor_duty[i] < 0:
                self.motor_duty[i] = 0

        # print(self.ref_height)


    def diff_eq(self,y):
        #姿勢位置
        v = np.array([y[7],y[8],y[9]],dtype=float)
        q4 = np.array([y[3],y[4],y[5],y[6]],dtype=float)
        rot = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=float)#重心姿勢(回転行列)[3x3]
        rot = self.q2c(q4)
   
        omg = np.array([y[0],y[1],y[2]],dtype=float)

        rm1 = np.array([ self.body_width/2, self.body_width/2,self.prop_offset])
        rm2 = np.array([ self.body_width/2,-self.body_width/2,self.prop_offset])
        rm3 = np.array([-self.body_width/2,-self.body_width/2,self.prop_offset])
        rm4 = np.array([-self.body_width/2, self.body_width/2,self.prop_offset])

        Fmg_xyz = np.dot(rot,np.array([0,0,-self.mass*self.grav],dtype=float))
        #制御量
        del1 = self.motor_duty[0]
        del2 = self.motor_duty[1]
        del3 = self.motor_duty[2]
        del4 = self.motor_duty[3]

        rm1_xyz = rm1
        norm1 = np.array([0,0,1])
        L1 = self.mf_coef * del1
        L1_xyz = L1*norm1
        M1_xyz = np.cross(rm1_xyz,L1_xyz)
        M1_z = np.array([0,0,-self.mm_coef * del1],dtype=float)

        rm2_xyz = rm2
        norm2 = np.array([0,0,1])
        L2 = self.mf_coef * del2
        L2_xyz = L2*norm2
        M2_xyz = np.cross(rm2_xyz,L2_xyz)
        M2_z =  np.array([0,0, self.mm_coef * del2],dtype=float)

        rm3_xyz = rm3
        norm3 = np.array([0,0,1])
        L3 = self.mf_coef * del3
        L3_xyz = L3*norm3
        M3_xyz = np.cross(rm3_xyz,L3_xyz)
        M3_z = np.array([0,0,-self.mm_coef * del3],dtype=float)

        rm4_xyz = rm4
        norm4 = np.array([0,0,1])
        L4 = self.mf_coef * del4
        L4_xyz = L4*norm4
        M4_xyz = np.cross(rm4_xyz,L4_xyz)
        M4_z = np.array([0,0, self.mm_coef * del4],dtype=float)

        M_xyz = M1_xyz + M2_xyz + M3_xyz + M4_xyz + M1_z + M2_z + M3_z + M4_z
        F_xyz = L1_xyz + L2_xyz + L3_xyz + L4_xyz + Fmg_xyz

        omgd = np.dot(np.linalg.inv(self.I),M_xyz - np.cross(omg,np.dot(self.I,omg)))
        qd4 = self.q2qd(q4,omg)
        state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=float)#状態変数

        state[0] = omgd[0]
        state[1] = omgd[1]
        state[2] = omgd[2]

        state[3] = qd4[0]
        state[4] = qd4[1]
        state[5] = qd4[2]
        state[6] = qd4[3]
 
        Ud = F_xyz/self.mass - np.array([omg[1]*v[2] - omg[2]*v[1],omg[2]*v[0] - omg[0]*v[2],omg[0]*v[1] - omg[1]*v[0]],dtype=float)
        state[7] = Ud[0]
        state[8] = Ud[1]
        state[9] = Ud[2]

        Xd = np.dot(rot.T,v)
        state[10] = Xd[0]
        state[11] = Xd[1]
        state[12] = Xd[2]
        return state

    def runge_kutta(self,f,y,dt):
        k1 = f(y)
        k2 = f(y+dt/2*k1)
        k3 = f(y+dt/2*k2)
        k4 = f(y+dt*k3)
        out = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        return out
    
    def e2c(self,zyx_euler):
        sr = np.sin(zyx_euler[0])
        sp = np.sin(zyx_euler[1])
        sy = np.sin(zyx_euler[2])
        cr = np.cos(zyx_euler[0])
        cp = np.cos(zyx_euler[1])
        cy = np.cos(zyx_euler[2])
        return np.array([[cp*cy,cp*sy,-sp],
                         [sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,sr*cp],
                         [cr*sp*cy+sr*sy,cr*sp*sy-sr*cy,cr*cp]],dtype=float)

    def c2e(self,rot):
        if 1-np.abs(rot[0,2]) > 2*10**(-4):
            roll = np.arctan(rot[1,2]/rot[2,2])
            pitch = np.arcsin(-rot[0,2])
            yaw = np.arctan2(rot[0,1],rot[0,0])
        else:
            roll = 0
            pitch = np.arcsin(-rot[0,2])
            yaw = np.arctan2(-rot[1,0],rot[1,1])
  
        return np.array([roll,pitch,yaw],dtype=float)

    def c2q(self,rot):
        q = np.array([0,0,0,1/2*np.sqrt(1+rot[0,0]+rot[1,1]+rot[2,2])],dtype=float)     
        q[0] = (rot[1,2]-rot[2,1])/(4*q[3])
        q[1] = (rot[2,0]-rot[0,2])/(4*q[3])
        q[2] = (rot[0,1]-rot[1,0])/(4*q[3])
  
        return q
    
    def q2c(self,q):
        rot = np.array([[q[0]**2-q[1]**2-q[2]**2+q[3]**2, 2*(q[0]*q[1]+q[2]*q[3]), 2*(q[0]*q[2]-q[1]*q[3])],
                      [2*(q[0]*q[1]-q[2]*q[3]), -q[0]**2+q[1]**2-q[2]**2+q[3]**2, 2*(q[1]*q[2]+q[0]*q[3])],
                      [2*(q[0]*q[2]+q[1]*q[3]), 2*(q[1]*q[2]-q[0]*q[3]), -q[0]**2-q[1]**2+q[2]**2+q[3]**2]],dtype=float)
        return rot

    
    def qdot(self,q,p):
        q_ans = np.array([0,0,0,0],dtype=float)
        q_ans[0] = q[0]*p[3] + q[3]*p[0] - q[2]*p[1] + q[1]*p[2]
        q_ans[1] = q[1]*p[3] + q[2]*p[0] + q[3]*p[1] - q[0]*p[2]
        q_ans[2] = q[2]*p[3] - q[1]*p[0] + q[0]*p[1] + q[3]*p[2]
        q_ans[3] = q[3]*p[3] - q[0]*p[0] - q[1]*p[1] - q[2]*p[2]
        return q_ans
    
    def q2qd(self,q,omg):
        qd = 0.5*np.array([[0,omg[2],-omg[1],omg[0]],
                           [-omg[2],0,omg[0],omg[1]],
                           [omg[1],-omg[0],0,omg[2]],
                           [-omg[0],-omg[1],-omg[2],0]])
        qd = np.dot(qd,q)
        return qd

class Window(QMainWindow):
    mesh0 = None
    mesh1 = None
    mesh2 = None
    mesh3 = None
    mesh4 = None

    ground = None
    scttrPlt = None
    scttrHis = None

    scatter_cycle = 0
    scatter_period = 10
    scatter_data = []

    region_size = 10000
    region_space = 500
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(0, 0, 1440, 1080) 
        self.setAcceptDrops(True)

        self.initUI()

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)

        layout = QVBoxLayout()
        centerWidget.setLayout(layout)

        self.viewer = gl.GLViewWidget()
        self.viewer.setBackgroundColor("k")
        layout.addWidget(self.viewer, 1)

        self.viewer.setWindowTitle('STL Viewer')
        self.viewer.setCameraPosition(distance=4000)
        self.viewer.orbit(45+90, -20)

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
        points0, faces0 = self.loadSTL('drone_body.stl')
        points1, faces1 = self.loadSTL('prop.stl')
        points2, faces2 = self.loadSTL('prop_ccw.stl')
        points3, faces3 = self.loadSTL('prop.stl')
        points4, faces4 = self.loadSTL('prop_ccw.stl')
        meshdata0 = gl.MeshData(vertexes=points0, faces=faces0)
        meshdata1 = gl.MeshData(vertexes=points1, faces=faces1)
        meshdata2 = gl.MeshData(vertexes=points2, faces=faces2)
        meshdata3 = gl.MeshData(vertexes=points3, faces=faces3)
        meshdata4 = gl.MeshData(vertexes=points4, faces=faces4)
        
        self.mesh0 = gl.GLMeshItem(meshdata=meshdata0,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 1, 1, 1))
        self.mesh1 = gl.GLMeshItem(meshdata=meshdata1,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 0, 0, 1))
        self.mesh2 = gl.GLMeshItem(meshdata=meshdata2,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(1, 0, 0, 1))
        self.mesh3 = gl.GLMeshItem(meshdata=meshdata3,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 0, 1, 1))
        self.mesh4 = gl.GLMeshItem(meshdata=meshdata4,shader='shaded', smooth=True, drawFaces=True, drawEdges=False, color=(0, 0, 1, 1))

        tmp = np.linspace(-self.region_size/2, self.region_size/2, int(self.region_size/2/1000))
        tmp_grid,empty  = np.meshgrid(tmp, tmp)
        z = np.zeros([tmp_grid.shape[0],tmp_grid.shape[1]])
        self.ground = gl.GLSurfacePlotItem(x=tmp, y=tmp, z=z, color=(0.0, 0.0, 0.0, 1))
        
        self.scttrPlt = gl.GLScatterPlotItem(pos=np.array([0,0,0]), size=10, color=(1.0, 0.0, 0.0, 1), pxMode=False)
        self.scttrHis = gl.GLScatterPlotItem(pos=np.array([0,0,0]), size=10, color=(1.0, 0.0, 0.0, 1), pxMode=False)

        self.viewer.addItem(self.ground)
        self.viewer.addItem(self.scttrPlt)
        self.viewer.addItem(self.scttrHis)
        self.viewer.addItem(self.mesh0)
        self.viewer.addItem(self.mesh1)
        self.viewer.addItem(self.mesh2)
        self.viewer.addItem(self.mesh3)
        self.viewer.addItem(self.mesh4)


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
        self.mesh1.translate( body_pos[0] + 35.3553, body_pos[1] + 35.3553, body_pos[2] + 15,True)
        self.mesh2.translate( body_pos[0] + 35.3553, body_pos[1] - 35.3553, body_pos[2] + 15,True)
        self.mesh3.translate( body_pos[0] - 35.3553, body_pos[1] - 35.3553, body_pos[2] + 15,True)
        self.mesh4.translate( body_pos[0] - 35.3553, body_pos[1] + 35.3553, body_pos[2] + 15,True)

        self.mesh1.rotate(prop_pos[0],0,0,1,True)
        self.mesh2.rotate(prop_pos[1],0,0,1,True)
        self.mesh3.rotate(prop_pos[2],0,0,1,True)
        self.mesh4.rotate(prop_pos[3],0,0,1,True)

def update():
    global window ,drone0
    t = time.time() - init_time
    print(t)
    drone0.update()
    body_pos = np.array([drone0.pos[0]*1000,drone0.pos[1]*1000,drone0.pos[2]*1000])#mm
    body_ori = np.array([drone0.ori[0]*180/np.pi,drone0.ori[1]*180/np.pi,drone0.ori[2]*180/np.pi])#deg
    prop_pos = np.array([drone0.prop_w[0]*t,-drone0.prop_w[1]*t,drone0.prop_w[2]*t,-drone0.prop_w[3]*t])
    window.moveMesh(body_pos,body_ori,prop_pos)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = Window()
    window.show()
    drone0 = drone()
    t1 = QTimer()
    t1.setInterval(10)
    t1.timeout.connect(update)
    t1.start()
    sys.exit(app.exec_())


