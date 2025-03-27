import numpy as np

class Camera:
    def __init__(self):
        self.rot_matrix = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.pos_matrix = np.matrix([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0],[0,0,-6.0,1.0]])
        self.zoom_matrix = np.matrix([[0.01,0,0,0],[0,0.01,0,0],[0,0,0.01,0],[0,0,0,1.0]])
        self.angles = np.array([-0.4,0.0,0.0])
        self.fixed_angle = False
        self.rotate(0, 0, 0)
        return

    def unfix(self):
        self.fixed_angle = False
        self.rotate(0, 0, 0)
        return

    def fix(self,x,y,z):
        self.fixed_angle = (x,y,z)
        self.rot_matrix = np.matrix(
            [[np.cos(y), 0, np.sin(y), 0], [0, 1, 0, 0], [-np.sin(y), 0, np.cos(y), 0], [0, 0, 0, 1]]) @ np.matrix(
            [[1, 0, 0, 0], [0, np.cos(x), -np.sin(x), 0], [0, np.sin(x), np.cos(x), 0], [0, 0, 0, 1]]) @ np.matrix(
            [[np.cos(z), -np.sin(z), 0, 0], [np.sin(z), np.cos(z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return

    def rotate(self,x,y,z):
        if(not self.fixed_angle):
            self.angles -= np.array([x,y,z])
            if(self.angles[0]>np.pi): self.angles[0]-= 2*np.pi
            if (self.angles[0] < -np.pi): self.angles[0] += 2 * np.pi
            x,y,z = self.angles
            self.rot_matrix = np.matrix([[np.cos(y),0,np.sin(y),0],[0,1,0,0],[-np.sin(y),0,np.cos(y),0],[0,0,0,1]])  @ np.matrix([[1,0,0,0],[0,np.cos(x),-np.sin(x),0],[0,np.sin(x),np.cos(x),0],[0,0,0,1]])@ np.matrix([[np.cos(z),-np.sin(z),0,0],[np.sin(z),np.cos(z),0,0],[0,0,1,0],[0,0,0,1]])
        return

    def translate(self,x,y,z):
        self.pos_matrix[3,0] += -x/400
        self.pos_matrix[3, 1] += y / 400
        return

    def zoom(self,zoom_value):
        self.zoom_matrix *= zoom_value
        self.pos_matrix[3, 0] *= zoom_value
        self.pos_matrix[3, 1] *= zoom_value
        self.zoom_matrix[3,3] = 1
        #print(self.zoom_matrix)
        return

    def get_view_matrix(self):
        return  self.rot_matrix @ self.zoom_matrix @ self.pos_matrix