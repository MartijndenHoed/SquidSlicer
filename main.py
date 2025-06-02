#This program was developed by Martijn den Hoed in the period of September 2024 to June 2025 to obtain a Master degree at the Delft technical university
#Most of the main slicing algorithm is documented, most of the other functions.
#In case you're interested in using or modifying (parts of) this code, feel free to contact me
import slice_engine as slicer
import stl
import pyglet as pg
import numpy as np
from scipy import ndimage
import camera as cam
import shaders_textured
import shaders_mono
import ui_setup
from tkinter import *
from tkinter.filedialog import askopenfilenames
from tkinter.filedialog import asksaveasfilename
import menus_setup
import os
import PIL
from functools import partial
import components
import pickle
import time
np.set_printoptions(suppress=True)
import json

class RenderGroupTextured(pg.graphics.Group):
    def __init__(self, texture, program,sharp=False, order=0, parent=None):
        super().__init__(order, parent)
        self.texture = texture
        self.program = program
        self.sharp = sharp

    def set_state(self):
        pg.gl.glActiveTexture(pg.gl.GL_TEXTURE0)
        pg.gl.glBindTexture(self.texture.target, self.texture.id)
        pg.gl.glEnable(pg.gl.GL_BLEND)
        if(self.sharp):
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_MAG_FILTER, pg.gl.GL_NEAREST)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_T, pg.gl.GL_CLAMP_TO_EDGE)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_S, pg.gl.GL_CLAMP_TO_EDGE)
        else:
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_MAG_FILTER, pg.gl.GL_LINEAR)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_T, pg.gl.GL_CLAMP_TO_BORDER)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_S, pg.gl.GL_CLAMP_TO_BORDER)


        pg.gl.glBlendFunc(pg.gl.GL_SRC_ALPHA, pg.gl.GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc(GL_ZERO, GL_SRC_ALPHA)
        self.program.use()

    def unset_state(self):
        pass
        #glDisable(GL_BLEND)

    def __hash__(self):
        return hash((self.texture.target, self.texture.id, self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)

class RenderGroupPlain(pg.graphics.Group):
    def __init__(self, program,sharp=False, order=0, parent=None):
        super().__init__(order, parent)
        self.program = program
        self.sharp = sharp

    def set_state(self):
        pg.gl.glEnable(pg.gl.GL_BLEND)
        if(self.sharp):
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_MAG_FILTER, pg.gl.GL_NEAREST)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_T, pg.gl.GL_CLAMP_TO_EDGE)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_S, pg.gl.GL_CLAMP_TO_EDGE)
        else:
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_MAG_FILTER, pg.gl.GL_LINEAR)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_T, pg.gl.GL_CLAMP_TO_BORDER)
            pg.gl.glTexParameterf(pg.gl.GL_TEXTURE_2D, pg.gl.GL_TEXTURE_WRAP_S, pg.gl.GL_CLAMP_TO_BORDER)


        pg.gl.glBlendFunc(pg.gl.GL_SRC_ALPHA, pg.gl.GL_ONE_MINUS_SRC_ALPHA)
        #glBlendFunc(GL_ZERO, GL_SRC_ALPHA)
        self.program.use()

    def unset_state(self):
        pass
        #glDisable(GL_BLEND)

    def __hash__(self):
        return hash(( self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)

class Object_save_template():
    def __init__(self,model):
        self.filename = model.filename
        self.secondary_models = model.secondary_models

        return



class Object():
    def __init__(self):
        self.filename = ""
        self.batch_stl = pg.graphics.Batch()
        self.secondary_models = []
        self.secondary_models_dims = []
        self.secondary_stl_batches = []
        self.visible = False
        self.render_mode = "stl"
        self.dims = ((0, 0), (0, 0), (0, 0))
        self.dims_extended = ((0, 0), (0, 0), (0, 0))
        self.offsets = (0, 0, 0)
        self.activelayer_height = 0
        self.layers = []
        self.circuit = Circuit()
        self.slicing_data = {
            "support_memory": None,
            "structural_mask": None,
            "support_spacing": None,
            "slice_traces": False,
            "traces_print_height": 10,
            "trace_layer": 0,
            "circuit_layer_index": 0,
            "layer_height": 0,
            "sub_renders": 10,
            "DPI": 0,
            "base_color": (255, 180, 0,40), #render edit2
 #           "secondary_color": (20, 255, 127, 42),
            "secondary_color": (255, 180, 0,40), #render edit
            "silver_color":  (200, 200, 200,255),
            "component_color": (30,30,30,255),
            "support_color": (50,200,200,15),
            "support_generation": False,
            "trace_slicing": False,
            "trace_width": 1,
            "colors": None,
            "color_count":0,
            "printhead_UV_offset":0,
            "render_components": True,
            "resolution": (0,0,0),
        }

    def clear(self):
        self.filename = ""
        self.secondary_models = []
        self.secondary_stl_batches = []
        del self.batch_stl
        self.batch_stl = pg.graphics.Batch()
        self.visible = False
        self.render_mode = "stl"
        self.dims = ((0, 0), (0, 0), (0, 0))
        self.dims_extended = ((0, 0), (0, 0), (0, 0))
        self.secondary_models_dims = []
        self.offsets = (0, 0, 0)
        self.activelayer_height = 0
        self.layers = []
        self.circuit = Circuit()

    def load_stl(self,filename):
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_WAIT))
        self.filename = filename
        self.visible = True
        self.render_mode = "stl"
        del self.batch_stl
        self.batch_stl = pg.graphics.Batch()
        stl_pic = pg.image.load('stl_tex.png')
        stl_tex = stl_pic.get_texture()
        self.RenderGroup_stl = RenderGroupPlain( mono_shader_program, sharp=True)

        mesh = stl.mesh.Mesh.from_file(self.filename)
        self.dims = slicer.get_model_dims(mesh)
        self.dims_extended = slicer.get_model_dims(mesh)
        vertices = []
        indices = []
        normals = []
        for i in range(0, len(mesh.v0)):
            vertices += mesh.v0[i].tolist()
            vertices += mesh.v2[i].tolist()
            vertices += mesh.v1[i].tolist()

            indices.append(i * 3 + 0)
            indices.append(i * 3 + 1)
            indices.append(i * 3 + 2)
            normal = unit_vector(mesh.normals[i])/2 + 0.5
            normals += (np.tile(normal,3)).tolist()
        vertex_list = mono_shader_program.vertex_list_indexed(len(vertices) // 3, pg.gl.GL_TRIANGLES, indices,
                                                                 self.batch_stl,
                                                                 self.RenderGroup_stl,
                                                                 position=('f', vertices),
                                                                 normals=('f', normals))
        pg.model.Model([vertex_list], [self.RenderGroup_stl], self.batch_stl)
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_DEFAULT))

    def load_secondary_stl(self,filename):
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_WAIT))
        self.render_mode = "stl"
        self.secondary_stl_batches.append(pg.graphics.Batch())
        stl_pic = pg.image.load('stl_tex_sec.png')
        stl_tex = stl_pic.get_texture()
        #self.RenderGroup_stl = RenderGroupTextured(stl_tex, texture_shader_program, sharp=False)

        mesh = stl.mesh.Mesh.from_file(filename)
        vertices = []
        indices = []
        normals = []
        dims = slicer.get_model_dims(mesh)
        self.secondary_models_dims.append(dims)
        self.dims_extended = ((np.min([self.dims[0][0],dims[0][0]]),np.max([self.dims[0][1],dims[0][1]])), (np.min([self.dims[1][0],dims[1][0]]),np.max([self.dims[1][1],dims[1][1]])), (np.min([self.dims[2][0],dims[2][0]]),np.max([self.dims[2][1],dims[2][1]])) )


        for i in range(0, len(mesh.v0)):
            vertices += mesh.v0[i].tolist()
            vertices += mesh.v2[i].tolist()
            vertices += mesh.v1[i].tolist()

            indices.append(i * 3 + 0)
            indices.append(i * 3 + 1)
            indices.append(i * 3 + 2)
            normal = unit_vector(mesh.normals[i])/2 + 0.5
            normals += (np.tile(normal,3)).tolist()
        vertex_list = mono_shader_program.vertex_list_indexed(len(vertices) // 3, pg.gl.GL_TRIANGLES, indices,
                                                                 self.secondary_stl_batches[-1],
                                                                 self.RenderGroup_stl,
                                                                 position=('f', vertices),
                                                                 normals=('f', normals))
        pg.model.Model([vertex_list], [self.RenderGroup_stl], self.secondary_stl_batches[-1])
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_DEFAULT))
        return

    def add_secondary_model(self,filename):
        self.secondary_models.append(filename)
        self.load_secondary_stl(filename)

        return

    def slice(self,export=False):
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_WAIT))
        if(not export):
            self.layers = []
        mesh = stl.mesh.Mesh.from_file(self.filename)
        secondary_meshes = []
        if (self.slicing_data["support_generation"]):
            dims = slicer.get_model_dims_with_support(mesh,1.0)
            print(dims)
            print( self.dims_extended)
            self.dims_extended = ((np.min([self.dims[0][0], dims[0][0]]), np.max([self.dims[0][1], dims[0][1]])),
                              (np.min([self.dims[1][0], dims[1][0]]), np.max([self.dims[1][1], dims[1][1]])),
                              (np.min([self.dims[2][0], dims[2][0]]), np.max([self.dims[2][1], dims[2][1]])))


        for secondary_model in self.secondary_models:
            secondary_meshes.append(stl.mesh.Mesh.from_file(secondary_model))
            secondary_meshes[-1].rotate([0.5,0,0],np.radians(-90))
        mesh.rotate([0.5,0,0],np.radians(-90))
        self.slicing_data["colors"] = np.asarray(256* [(0,0,0,0)])
        self.slicing_data["color_count"] = 2
        self.slicing_data["colors"][0] = self.slicing_data["base_color"]
        self.slicing_data["colors"][1] = self.slicing_data["secondary_color"]
        self.slicing_data["colors"][2] = self.slicing_data["support_color"]
        self.slicing_data["colors"][3] = self.slicing_data["component_color"]
        self.slicing_data["colors"][4] = self.slicing_data["silver_color"]
        self.slicing_data["trace_layer"] = 1





        resolution = [int(round(( (self.dims_extended[0][1]-self.dims_extended[0][0]) /25.4)*self.slicing_data["DPI"])),int(round(((self.dims_extended[2][1]-self.dims_extended[2][0])/25.4)*self.slicing_data["DPI"]))]
        layer_count = int((self.dims_extended[1][1]-self.dims_extended[1][0])/self.slicing_data["layer_height"])
        self.slicing_data["resolution"] = (resolution[0], resolution[1], layer_count)
        sandwich_mode = False #sandwich mode indicates that there is a circuit layer sandwiched between two structural models
        sandwich_mode_top = 0 #this is the heighest point of the sandwiched circuit layer
        sandwich_slicing_mode = "combined"

        component_buffer = []

        for circuit_layer in self.circuit.circuit_layers: #check if there is a sandwiched layer
            if(circuit_layer.sandwich_mode):
                sandwich_mode = True
                sandwich_mode_top = max(sandwich_mode_top,circuit_layer.z_height)


        for via in self.circuit.vias:
            via.update_values()

        if(self.circuit.circuit_layers and self.slicing_data["slice_traces"]):
            for circuit_layer in self.circuit.circuit_layers:
                circuit_layer.generate_layer(render=False, resolution = resolution)
            circuit_layer = np.zeros((resolution[0],resolution[1]),dtype=np.uint8) #create some dummy layers
            circuit_layer_buffer = np.zeros((resolution[0],resolution[1]),dtype=np.uint8)
            circuit_layer_buffer_layer = layer_count-1 #keep track of the layer of the circuit layer buffer
            circuit_via_layer_buffer = None

        for circuit_layer_object in self.circuit.circuit_layers:
            for component in circuit_layer_object.components:
                component_buffer.append(component)
                component.actual_z_level = circuit_layer_object.z_height
                component.slicing_activated = False


        for i in reversed(range(0,layer_count)): #slice from top to bottom
            print(f"Slicing: {int(100*(1-i/layer_count))}%")
            z_level = self.slicing_data["layer_height"] * i + 0.5*self.slicing_data["layer_height"] #slice the middle of the layer

            layer_array = np.zeros(resolution) #layer_array is the array where the current layer data is to be stored

            layer_array_main = slicer.slice_model_2d(mesh,z_level+self.dims_extended[1][0],self.slicing_data["DPI"]) #this is the layer array of only the main part (for sandwich slicing)
            layer_array_secondaries = [] #these are the layer arrays of the secondary parts (for sandwich slicing)

            main_model_layer_offset = (((np.asarray((self.dims_extended[0][1]-self.dims[0][1],self.dims_extended[2][1]-self.dims[2][1]))/25.4)*self.slicing_data["DPI"])).astype(np.int64) #calculate the position of the main part in the full object
            np.copyto(layer_array[main_model_layer_offset[0]:main_model_layer_offset[0]+layer_array_main.shape[0],main_model_layer_offset[1]:main_model_layer_offset[1]+layer_array_main.shape[1]],layer_array_main,where=(layer_array_main!=0)) #copy the layer array of the main part into the full object

            if(sandwich_slicing_mode=="combined" and z_level<sandwich_mode_top): #Check if we are in combined mode but below the top of the sandwiched circuit layer
                sandwich_slicing_mode = "separated" #stop combining secondary  models when we reach the sandwiched circuit, slice the rest of the secondary models as fillup layers

                lowest_point_secondary_model = self.dims_extended[1][1] #Check what the lowest point is of all secondary parts, so we can slice only up until the lowest part
                for secondary_model_dims in self.secondary_models_dims:
                    lowest_point_secondary_model = np.min((lowest_point_secondary_model,secondary_model_dims[1][0]))


                for l in reversed(range(int((lowest_point_secondary_model-self.dims_extended[1][0])/self.slicing_data["layer_height"]),i+1)): #loop through the remaining section of the secondary parts that are to be printed seperately
                    #This is the subroutinge for slicing the layers of the secondary parts that are printed seperately (over the sandwiched circuit layer)
                    z_level_new = self.slicing_data["layer_height"] * l + 0.5*self.slicing_data["layer_height"] #Determine the z-level of the current layer
                    layer_array_new = np.zeros(resolution) #Temporary new layer array for this subroutine
                    layer_array_secondaries = []

                    #This part loops through all secondary parts, slices them at the current level, and places them in the current layer array
                    for k in range(0, len(self.secondary_models)):
                        layer_array_secondaries.append(
                            slicer.slice_model_2d(secondary_meshes[k], z_level_new + self.dims_extended[1][0],
                                                  self.slicing_data["DPI"])) #Slice the secondary part

                        # secondary_model_layer_offset = (((np.asarray(
                        #     (self.dims_extended[0][0] - self.secondary_models_dims[k][0][0],
                        #      self.dims_extended[2][1] - self.secondary_models_dims[k][2][1])) / 25.4) *
                        #                                  self.slicing_data["DPI"])).astype(np.int64) #Calculate where in the layer array the secondary part is placed
                        secondary_model_layer_offset = (((np.asarray(
                            (self.secondary_models_dims[k][0][0] - self.dims_extended[0][0],
                             self.secondary_models_dims[k][2][1] - self.dims_extended[2][1])) / 25.4) *
                                                         self.slicing_data["DPI"])).astype(np.int64)


                        np.copyto(layer_array_new[secondary_model_layer_offset[0]:secondary_model_layer_offset[0] +
                                                                              layer_array_secondaries[k].shape[0],
                                  secondary_model_layer_offset[1]:secondary_model_layer_offset[1] +
                                                                  layer_array_secondaries[k].shape[1]],
                                  2 * layer_array_secondaries[k], where=(layer_array_secondaries[k] != 0)) #Place the sliced secondary part in the layer array, at the correct position

                    print_level = z_level + self.slicing_data["layer_height"] - self.slicing_data[
                        "traces_print_height"] * np.floor(
                        ((sandwich_mode_top - (z_level_new)) / self.slicing_data["traces_print_height"])) #Determine the printheight of the current layer. As this is a fillup layer, it needs to be checked if the jetting height is not too high. If that's the case, it is printed in mutiple steps

                    #Render the layer if we're not in export mode
                    if (export == False):
                        layer = Sliced_layer(layer_array_new.astype(np.uint8), z_level_new, self.slicing_data["layer_height"],
                                             self.slicing_data["sub_renders"]
                                             , self.slicing_data["colors"],
                                             dims=(self.dims_extended[0], self.dims_extended[2], self.dims_extended[1]),
                                             type="struc",print_level=print_level)
                        self.layers.append(layer)

                    #Export the layer if we're in export mode. The name contains the index of the fillup layer, and the layer at which it should be printed
                    if(export):
                        print_layer = int(print_level/self.slicing_data["layer_height"])
                        self.export_layer(str(export) + f"/strucfill_{print_layer}_{l-int((lowest_point_secondary_model-self.dims_extended[1][0])/self.slicing_data['layer_height'])}.png",layer_array_new,UV_offset=self.slicing_data["printhead_UV_offset"])

            #If we don't have to seperate the secondary parts from the main part, we can just combine them into the same layer array; this is 'combined' mode
            if(sandwich_slicing_mode=="combined"):
                #Loop through all secondary parts, slice them, and add them to the current layer array:
                for k in range(0,len(self.secondary_models)):
                    layer_array_secondaries.append( slicer.slice_model_2d(secondary_meshes[k],z_level+self.dims_extended[1][0],self.slicing_data["DPI"]) ) #Slice the secondary part layer
                    secondary_model_layer_offset = (((np.asarray(
                        (self.secondary_models_dims[k][0][0]-self.dims_extended[0][0], self.secondary_models_dims[k][2][1]-self.dims_extended[2][1])) / 25.4) *
                                               self.slicing_data["DPI"])).astype(np.int64) #Calculate where in the layer array the secondary part layer should be
                    np.copyto(layer_array[secondary_model_layer_offset[0]:secondary_model_layer_offset[0] + layer_array_secondaries[k].shape[0],
                          secondary_model_layer_offset[1]:secondary_model_layer_offset[1] + layer_array_secondaries[k].shape[1]],
                          layer_array_secondaries[k], where=(layer_array_secondaries[k] != 0)) #Copy the sliced layer into the layer array



            via_fill_top_layer = None #this indicates if the top of a via fill is present in the layer
            for via in self.circuit.vias: #Loop through all vias
                if( not via.generate_structure_layer(z_level,self.slicing_data["DPI"]) is None): #Check if the via is present in the current layer

                    via_layer = via.generate_structure_layer(z_level,self.slicing_data["DPI"]) #Generate the (slice of the) via structure
                    via_pos = (int(via.position[0] * resolution[0]),int(via.position[1] * resolution[1])) #determine the position of the via in the layer array
                    via_width = int(np.round(0.5*via_layer.shape[0])) #calculate the width and length of the via structure (in terms of pixels)
                    via_length = int(np.round(0.5*via_layer.shape[1]))

                    start_row = via_pos[0] - via_width #Calculate at which indices the via should be placed in the layer array
                    start_col = via_pos[1] - via_length
                    end_row = start_row + via_layer.shape[0]
                    end_col = start_col + via_layer.shape[1]

                    clip_start_row = max(start_row, 0) #Clip these indices if the via is at the edge of the object
                    clip_start_col = max(start_col, 0)
                    clip_end_row = min(end_row, layer_array.shape[0])
                    clip_end_col = min(end_col, layer_array.shape[1])

                    via_start_row = clip_start_row - start_row #The indices for the via structure also needs to be clipped
                    via_start_col = clip_start_col - start_col
                    via_end_row = via_start_row + (clip_end_row - clip_start_row)
                    via_end_col = via_start_col + (clip_end_col - clip_start_col)

                    #layer_array[clip_start_row:clip_end_row, clip_start_col:clip_end_col] = \
                     #   via_layer[via_start_row:via_end_row, via_start_col:via_end_col].astype(np.uint8) #Place the via structure in the layer array
                    np.copyto(layer_array[clip_start_row:clip_end_row, clip_start_col:clip_end_col], via_layer[via_start_row:via_end_row, via_start_col:via_end_col],
                              where=(via_layer[via_start_row:via_end_row, via_start_col:via_end_col]==0)) #Place the via structure in the layer array, but only the negative volume

                    if(via.new_step_flag): #For stepped vias, check if we have reached a new step
                        #via_fill_arr = via.generate_via_fill(z_level-self.slicing_data["layer_height"], self.slicing_data["DPI"],
                        #                                     self.slicing_data["layer_height"]) #generate the fillup layers to fil the new step
                        via_circuit_layer = via.generate_via_circuit_layer(self.slicing_data["DPI"]) #Generate the traces for the new step
#marker
                        via_fill_layer_count = int(via.step_height / self.slicing_data["layer_height"])
                        for j in range(0,via_fill_layer_count): #Loop through the fillup layers to render or export them
                            print(f"Via layer: {j}")
                            via_fill_layer_arr = np.zeros(layer_array.shape) #Generate a new layer the size of the full layer array


                            via_fill_layer = via.generate_via_fill_layer(z_level-self.slicing_data["layer_height"]*(via_fill_layer_count-j-1), self.slicing_data["DPI"],
                                                             self.slicing_data["layer_height"])
                            print(via_fill_layer.shape)
                            #via_fill_layer_arr[clip_start_row:clip_end_row, clip_start_col:clip_end_col] = \
                            #    via_fill_arr[via_start_row:via_end_row, via_start_col:via_end_col,j]# Place the fillup structure in the new layer array

                            via_fill_layer_arr[clip_start_row:clip_end_row, clip_start_col:clip_end_col] = \
                                via_fill_layer[via_start_row:via_end_row, via_start_col:via_end_col]# Place the fillup structure in the new layer array


                            if(not export): #Render the fillup layers
                                via_fill_layer = Sliced_layer((2*via_fill_layer_arr).astype(np.uint8),z_level-self.slicing_data["layer_height"]*(via_fill_layer_count-j-1),self.slicing_data["layer_height"],self.slicing_data["sub_renders"]
                                         ,self.slicing_data["colors"],dims=(self.dims_extended[0],self.dims_extended[2],self.dims_extended[1]),type="struc",print_level=z_level+self.slicing_data["layer_height"])
                                self.layers.append(via_fill_layer)
                        via.new_step_flag = False
                        via_fill_top_layer = via_fill_layer_arr #Store that there is a via fillup at this layer, this is needed so we know we can place a circuit layer over this fillup later

                        #To ensure that the circuit on the via is not segmented vertically (when other circuit layers are present in the vertical span of the via),
                        #we make a buffer were the specific via circuit is stored. Later on, in the circuit slicing functions, we ensure that this buffer is exported without being segmented
                        if(circuit_via_layer_buffer is None): circuit_via_layer_buffer = np.zeros(resolution) #Make a layer for the circuit layer that is placed on the step
                        circuit_via_layer_buffer[clip_start_row:clip_end_row, clip_start_col:clip_end_col] = \
                            via_circuit_layer[via_start_row:via_end_row, via_start_col:via_end_col].astype(np.uint8)
                        #Place the via circuit in the buffer

                        circuit_layer[clip_start_row:clip_end_row, clip_start_col:clip_end_col] = \
                            via_circuit_layer[via_start_row:via_end_row, via_start_col:via_end_col].astype(np.uint8)
                        #Also store it in the regular circuit layer

                        self.slicing_data["circuit_layer_index"] = self.slicing_data["traces_print_height"]/self.slicing_data["layer_height"] #Store that we added to the circuit layer buffer here, this ensures that the circuit layer buffer is printed at this layer

            #This section loops through all components and checks if they are present in the current layer
            for component in component_buffer: #loop through all components
                if(not component.slicing_activated): #Check if they are not yet activated
                    if((component.buried and z_level<=component.actual_z_level  and layer_array[int(component.position[0] * resolution[0]),int(component.position[1] * resolution[1])]) or \
                            (not component.buried and z_level<=component.actual_z_level + component.height)): #Check of they are present at or below the current height, and if there is actually material present here to place them on
                        component.slicing_activated = True #activate the component, this is just a flag so that we only the the actual height once
                        component.actual_z_level = z_level #set the actual z level, this can differ from the initial component z level if it is placed floating in the air
                        if(not component.buried): component.actual_z_level = np.max((component.actual_z_level,component.parent_z_level+component.height))
                        if (export == False and self.slicing_data["render_components"]): #render the component if we're not in export mode
                            self.generate_component_layers(component)

            for component in component_buffer: #Loop again through all components, here we generate the component cavity
                if(z_level<=(component.actual_z_level) and z_level>= (component.actual_z_level - component.height)): #Check if the component intersects the current layer
                    component_current_z_level = z_level - (component.actual_z_level - component.height) #Calculate at which z level in the component we are, in the coordinate system of the component
                    component_layer_array = component.generate_hole_layer_array(component_current_z_level,(self.dims[0], self.dims[2], self.dims[1]),
                                                                           resolution) #Generate the cavity structure
                    layer_array = np.logical_and(np.logical_xor(layer_array,component_layer_array),layer_array) #Place the cavity in the layer array

            #All the operations on the layer array are now performed, so we can either render the layer or export it
            if(export==False):
                layer = Sliced_layer(layer_array.astype(np.uint8),z_level,self.slicing_data["layer_height"],self.slicing_data["sub_renders"]
                                     ,self.slicing_data["colors"],dims=(self.dims_extended[0],self.dims_extended[2],self.dims_extended[1]),type="struc")
                self.layers.append(layer)
            else:
                self.export_layer(str(export) + f"/struc_{i}.png",layer_array,UV_offset=self.slicing_data["printhead_UV_offset"])

            #The following section concerns the slicing of circuit layers
            if (self.circuit.circuit_layers and self.slicing_data["slice_traces"]): #only do the following stuff if there actually are circuit layers and if we have to slice them
                self.slicing_data["circuit_layer_index"] += 1 #keep track that we moved down a layer

                for circuit_layer_object in self.circuit.circuit_layers: #Loop through all circuit layers
                    if ((circuit_layer_object.z_height > z_level  and circuit_layer_object.z_height <= z_level+ self.slicing_data["layer_height"]) or (i==layer_count-1 and circuit_layer_object.z_height > z_level)): #check if there is a circuit layer here or if we are at the ground
                        if(via_fill_top_layer is not None):
                            via_fill_top_layer_circuit_arr = np.logical_and(via_fill_top_layer,circuit_layer_object.layer_array)
                            self.layers.append(Sliced_layer(
                                (via_fill_top_layer_circuit_arr * 5).astype(np.uint8),
                                z_level + 0.9 * self.slicing_data["layer_height"], 0.1,
                                self.slicing_data["sub_renders"], self.slicing_data["colors"],
                                dims=(self.dims[0], self.dims[2], self.dims[1]), type="via_top_circ",
                                print_level=z_level+self.slicing_data["layer_height"]))
                        else:
                            via_fill_top_layer_circuit_arr = np.zeros(circuit_layer_object.layer_array.shape)

                        circuit_layer = np.logical_or(np.logical_xor(circuit_layer_object.layer_array,via_fill_top_layer_circuit_arr), circuit_layer).astype(np.uint8)
                        self.slicing_data["circuit_layer_index"] = self.slicing_data["traces_print_height"]/self.slicing_data["layer_height"]

                if (self.slicing_data["circuit_layer_index"] * self.slicing_data["layer_height"] >= self.slicing_data["traces_print_height"] or i==0): #This triggers if a new segment of circuit has to be printed
                    self.slicing_data["circuit_layer_index"] = 0
                    self.slicing_data["trace_layer"] += 1
                    #self.slicing_data["colors"][self.slicing_data['trace_layer'] + 2] = (255, 150 + self.slicing_data["trace_layer"] * 60, 150+ self.slicing_data["trace_layer"] * 180, 150+ self.slicing_data["trace_layer"] * 100)
                    self.slicing_data["colors"][self.slicing_data['trace_layer'] + 2] = self.slicing_data["silver_color"]
                    # if(self.slicing_data['trace_layer'] ==2): self.slicing_data["colors"][self.slicing_data['trace_layer'] + 2] = (255, 153, 255, 85)
                    # if (self.slicing_data['trace_layer'] == 3): self.slicing_data["colors"][
                    #     self.slicing_data['trace_layer'] + 2] = (255, 255, 85, 85) #render_edit

                    if(export):
                        self.export_layer(str(export) + f"/circ_{circuit_layer_buffer_layer}.png", circuit_layer_buffer)
                    circuit_layer_buffer = np.zeros((resolution[0],resolution[1]), dtype=np.uint8)
                    circuit_layer_buffer_layer = i


                circuit_layer_segment = np.logical_and(layer_array, circuit_layer)
                circuit_layer -= circuit_layer_segment.astype(np.uint8)
                if(circuit_via_layer_buffer is not None):
                    circuit_layer_buffer += (circuit_via_layer_buffer).astype(np.uint8)
                    circuit_via_layer_buffer = None
                circuit_layer_buffer += circuit_layer_segment
                if(export==False):
                    if (i > 0):
                        self.layers.append(Sliced_layer((circuit_layer_segment * (self.slicing_data["trace_layer"] + 3)).astype(np.uint8),
                                                        z_level + 0.9 * self.slicing_data["layer_height"], 0.1, self.slicing_data["sub_renders"], self.slicing_data["colors"],
                                                        dims=(self.dims_extended[0],self.dims_extended[2],self.dims_extended[1]),type="circ",print_level = (circuit_layer_buffer_layer+1) * self.slicing_data["layer_height"]))
                    else:
                        self.layers.append(
                            Sliced_layer((circuit_layer * 5).astype(np.uint8), z_level, 0.1, self.slicing_data["sub_renders"], self.slicing_data["colors"],
                                         dims=(self.dims_extended[0],self.dims_extended[2],self.dims_extended[1]),type="circ",print_level = (circuit_layer_buffer_layer+1) * self.slicing_data["layer_height"]))




            if (self.slicing_data["support_generation"]):
                support_layer_array = 3*self.generate_support_layer(layer_array)
                if (self.slicing_data["structural_mask"] is None):
                    self.slicing_data["structural_mask"] = self.generate_support_structural_mask(support_layer_array.shape)
                support_layer_array = support_layer_array - np.logical_and(support_layer_array,self.slicing_data["structural_mask"])
                support_layer = Sliced_layer(support_layer_array, z_level, self.slicing_data["layer_height"], self.slicing_data["sub_renders"], self.slicing_data["colors"],
                                     dims=(self.dims_extended[0],self.dims_extended[2],self.dims_extended[1]),type="supp")
                self.layers.append(support_layer)
        layer_count * self.slicing_data["layer_height"]
        self.visible=True
        if(not export):
            self.render_mode = "sliced"
        self.slicing_data["support_memory"] = None
        self.slicing_data["structural_mask"] = None
        if ( export):
            print("File Sliced Succesfully!")
        self.layers.sort(key=lambda x: x.z_level)

        self.activelayer_height =  model.layers[-1].z_level + model.slicing_data["layer_height"]
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_DEFAULT))
        return

    def export_layer(self,export_name,layer_array,UV_offset=0):
        UV_offset_arr = np.zeros(
            (int(UV_offset * self.slicing_data["DPI"] / 25.4), layer_array.shape[1]))
        full_arr = np.concatenate((UV_offset_arr, layer_array), 0)
        img_data = PIL.Image.fromarray((np.logical_not(full_arr) * 255).astype(np.uint8))
        img_data.save(export_name)

    def generate_support_layer(self,layer_array):
        height_limit = 1000000

        if(self.slicing_data["support_memory"] is None):
            self.slicing_data["support_memory"] = layer_array.astype(np.int64)
            return np.zeros((layer_array.shape[0],layer_array.shape[1]))


        new_material = np.logical_and(np.logical_not(self.slicing_data["support_memory"]),layer_array)
        #new_material_dilated = ndimage.binary_dilation(new_material,iterations = 2)
        #new_material_dilated = np.logical_and(np.logical_not(self.slicing_data["support_memory"]),new_material_dilated)


        self.slicing_data["support_memory"][self.slicing_data["support_memory"] != 0] += 1
        np.copyto(self.slicing_data["support_memory"],new_material,where=(new_material==True))


        mask = (self.slicing_data["support_memory"] == height_limit)

        # Step 2: Dilate the mask
        inflated_mask = ndimage.binary_dilation(mask)
        np.copyto(self.slicing_data["support_memory"], inflated_mask, where=(self.slicing_data["support_memory"] == 0))

        #support_layer = (self.slicing_data["support_memory"]!=0).astype(np.uint8)
        #print(self.slicing_data["support_memory"])
        support_layer =  np.logical_xor(self.slicing_data["support_memory"],layer_array) * np.logical_not(layer_array)
        #self.slicing_data["support_memory"] = np.logical_or(self.slicing_data["support_memory"],support_layer)
        #self.slicing_data["support_memory"] = np.logical_or(self.slicing_data["support_memory"], layer_array)
        return support_layer.astype(np.uint8)

    def generate_support_structural_mask(self,shape):
        mesh = np.mgrid[0:shape[0], 0: shape[1]]
        mask = np.logical_and(np.logical_not(mesh[0]%self.slicing_data["support_spacing"]),np.logical_not(mesh[1]%self.slicing_data["support_spacing"]))
        return mask

    def generate_component_layers(self,component):
        if(not component.component_rules[0]): return
        layers = int(np.round(component.height/self.slicing_data["layer_height"]))
        for i in range(0,layers):
            layer_array = component.generate_component_layer_array((self.dims[0],self.dims[2],self.dims[1]),self.slicing_data["resolution"])
            z_level = component.actual_z_level - i * self.slicing_data["layer_height"]
            print_level = component.parent_z_level
            layer = Sliced_layer((4*layer_array).astype(np.uint8), z_level, self.slicing_data["layer_height"],
                                 self.slicing_data["sub_renders"]
                                 , self.slicing_data["colors"]
                                 ,print_level = print_level,
                                 dims=(self.dims_extended[0], self.dims_extended[2], self.dims_extended[1]),
                                 type="comp")
            self.layers.append(layer)

    def draw(self,camera):
        mono_shader_program['offset'] = pg.math.Mat4.from_translation(
            pg.math.Vec3(self.offsets[0], self.offsets[1],
                         self.offsets[2]))
        if(self.render_mode=="stl" and self.visible):
            pg.gl.glEnable(pg.gl.GL_DEPTH_TEST)
            mono_shader_program['color_in'] = (self.slicing_data["base_color"][0]/255.0,self.slicing_data["base_color"][1]/255.0,self.slicing_data["base_color"][2]/255.0)
            plate_batch.draw()
            self.batch_stl.draw()

            #The following enables wireframe rendering:
            #mono_shader_program['color_in'] = (0.1,0.1,0.1)
            #pg.gl.glPolygonMode(pg.gl.GL_FRONT_AND_BACK, pg.gl.GL_LINE)
            #self.batch_stl.draw()
            #pg.gl.glPolygonMode(pg.gl.GL_FRONT_AND_BACK, pg.gl.GL_FILL)

            mono_shader_program['color_in'] = (
                self.slicing_data["secondary_color"][0] / 255.0, self.slicing_data["secondary_color"][1] / 255.0,
                self.slicing_data["secondary_color"][2] / 255.0)
            for batch in self.secondary_stl_batches:
                batch.draw()
            pg.gl.glDisable(pg.gl.GL_DEPTH_TEST)

        texture_shader_program['offset'] = pg.math.Mat4.from_translation(
            pg.math.Vec3(self.offsets[0], self.offsets[1],
                         self.offsets[2]))
        if(self.render_mode=="sliced" and self.visible):

            if(camera.angles[0]<0):
                for layer in self.layers:
                    layer.draw()
                    if(layer.visible and not self.slicing_data["slice_traces"]):
                        for circuit_layer in self.circuit.circuit_layers:
                            if(circuit_layer.z_height>layer.z_level-layer.layer_height and circuit_layer.z_height<=layer.z_level):
                                circuit_layer.draw()

                if(not self.slicing_data["slice_traces"]): #render_edit
                    for circuit_layer in self.circuit.circuit_layers:
                        if(circuit_layer.z_height>=self.dims[1][1]-self.dims[1][0]):
                            circuit_layer.draw()
            else:
                for layer in reversed(self.layers):
                    if (not self.slicing_data["slice_traces"]):
                        for circuit_layer in self.circuit.circuit_layers:
                            #print(circuit_layer.z_height, self.dims[1])
                            if (circuit_layer.z_height >= self.dims[1][1] - self.dims[1][0]):
                                circuit_layer.draw()
                    if (layer.visible and not self.slicing_data["slice_traces"]):
                        for circuit_layer in self.circuit.circuit_layers:
                            if (circuit_layer.z_height > layer.z_level - layer.layer_height and circuit_layer.z_height <= layer.z_level):
                                circuit_layer.draw()
                    layer.draw()

        texture_shader_program['offset'] = pg.math.Mat4()
        mono_shader_program['offset'] = pg.math.Mat4()
        return



class Sliced_layer():

    def __init__(self,data,z_level,layer_height,sub_renders,colors,offset=(0,0,0),scale=1,dims=((1,-1),(1,-1),(1,-1)),sharp=False,type="struc",print_level=None):
        self.visible = True
        self.data = data
        self.type = type
        self.z_level = z_level + offset[2]
        if(print_level==None):
            self.print_level = z_level
        else:
            self.print_level = print_level
        self.layer_height = layer_height
        self.sub_renders = sub_renders
        self.colors = colors
        self.tex = self.render_texture()
        self.group = RenderGroupTextured(self.tex, texture_shader_program,sharp)
        self.batch = pg.graphics.Batch()
        self.rendered_layers = []
        for i in range(0,sub_renders):
            height = self.z_level + (layer_height/sub_renders)*i
            self.rendered_layers.append(self.create_quad(height, self.group, self.batch,offset,scale,dims))
        return

    def draw(self):
        if(self.visible):
            self.batch.draw()
        return

    def render_texture(self):
        image = pg.image.create(self.data.shape[0], self.data.shape[1])
        layer_texture = (np.kron(self.data.T, np.array((0, 0, 0, 0), dtype=np.uint8))).flatten()
        uniques = np.unique(self.data)
        uniques = uniques[uniques != 0]
        for i in uniques:
            layer_mask = (self.data == i)
            layer_texture += (np.kron(layer_mask.T, np.array(self.colors[i - 1], dtype=np.uint8))).flatten()
        image.set_data("RGBA", self.data.shape[0] * 4, layer_texture.tobytes())

        return image.get_texture()

    def create_quad(self,height, group, batch,offset=(0,0,0),scale=1,dims=((1,-1),(1,-1),(1,-1))):
        x_off,y_off,z_off = offset
        z_off = dims[2][0]
        vertices = [dims[0][0]*scale+x_off, height+z_off, dims[1][1]*scale+y_off, dims[0][1]*scale+x_off, height+z_off, dims[1][1]*scale+y_off, dims[0][0]*scale+x_off, height+z_off, dims[1][0]*scale+y_off, dims[0][1]*scale+x_off, height+z_off, dims[1][0]*scale+y_off]

        #vertices = [dims[0][0]*scale+x_off, height+z_off, -dims[1][1]*scale+y_off, dims[0][1]*scale+x_off, height+z_off, -dims[1][1]*scale+y_off, dims[0][0]*scale+x_off, height+z_off, -dims[1][0]*scale+y_off, dims[0][1]*scale+x_off, height+z_off, -dims[1][0]*scale+y_off]
        self.vertices = vertices
        indices = [1, 2, 0, 1, 3, 2]
        vertex_list = texture_shader_program.vertex_list_indexed(len(vertices) // 3, pg.gl.GL_TRIANGLES, indices, batch, group,
                                                 position=('f', vertices),
                                                 tex_coords=('f', (0, 0, 0, 1, 0, 0,   0, 1, 0,   1, 1, 0 )  ))
        return pg.model.Model([vertex_list], [group], batch)

class Via():
    def __init__(self,origin_layer,exit_layer,position,type="stairwell"):
        self.type = type
        self.position = position
        self.origin_layer = origin_layer
        self.exit_layer = exit_layer
        self.step_height = 1
        self.width = 4
        self.bottom_height = min(origin_layer.z_height,exit_layer.z_height)
        self.top_height = max(origin_layer.z_height, exit_layer.z_height)
        return

    def update_values(self):
        self.max_step_height = model.slicing_data["traces_print_height"]
        self.height = self.top_height - self.bottom_height
        if(self.type=="stairwell"):
            self.step_count = 2*np.ceil(self.height/(self.max_step_height*2))
        elif(self.type=="dimple"):
            self.step_count = 1
        print(self.type)
        print(f"steps: {self.step_count}")
        self.step_height = self.height/self.step_count
        self.slope = self.step_height/(self.width/2)
        self.current_direction = 0
        self.new_step_flag = False

    def generate_structure_layer(self,z_level,DPI):
        if(z_level>self.top_height or z_level<self.bottom_height):
            return None
        else:
            if(self.type=="stairwell"):
                current_step = np.floor(((z_level-self.bottom_height)/self.height)*self.step_count) #decide which stair of the staircase we're on, starting with 0

                arr_size = int(self.width * (DPI / 25.4)) #determine the size of the structure in pixels
                z = (z_level - self.bottom_height) / (self.top_height - self.bottom_height) #calculate the height of the current layer in mm, 0 is the floor of the via
                z_normalized = ((z_level - self.bottom_height)-np.floor(current_step/2)*2*self.step_height)/(2*self.step_height)
                if(current_step%2 == 0): #create the function for the stair in the right direction, in normalized coordinates
                    rule = f"(x-0.1)<-z"

                else:
                    rule = f"(x-0.1)>z-1"
                new_direction = current_step%2
                if(self.current_direction != new_direction):
                    self.new_step_flag = True
                    self.current_direction = new_direction

                arr = np.ones((arr_size, arr_size), dtype=np.uint8) #create a horizontal layer of the via structure
                #rule = f"np.sqrt(x**2 + y**2) > {0.5*DPI/25.4}"
                mesh = np.mgrid[-0.5*arr_size:0.5*arr_size, -0.5*arr_size:0.5*arr_size]
                rule = rule.replace('z', str(z_normalized))
                rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                arr *= eval(rule)
            if(self.type=="dimple"):
                arr_size = int(self.width * (DPI / 25.4)) #determine the size of the structure in pixels
                z = (z_level - self.bottom_height) / (self.top_height - self.bottom_height) #calculate the height of the current layer in mm, 0 is the floor of the via
                z_normalized = ((z_level - self.bottom_height)-self.step_height)/(self.step_height)
                hole_top = 0.41
                hole_bottom = 0.26
                #print(z_normalized)
                if(not self.current_direction):
                    self.current_direction = True
                    self.new_step_flag = True
                hole_size = hole_bottom + (1+z_normalized)*(hole_top-hole_bottom)

                rule = f"np.sqrt(x**2 + y**2) >= {hole_size}"

                arr = np.ones((arr_size, arr_size), dtype=np.uint8) #create a horizontal layer of the via structure
                #rule = f"np.sqrt(x**2 + y**2) > {0.5*DPI/25.4}"
                mesh = np.mgrid[-0.5*arr_size:0.5*arr_size, -0.5*arr_size:0.5*arr_size]
                rule = rule.replace('z', str(z_normalized))
                rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                arr *= eval(rule)

        return arr

    def generate_via_circuit_layer(self,DPI):
        arr_size = int(self.width * (DPI / 25.4))  # determine the size of the structure in pixels
        arr = np.ones((arr_size, arr_size), dtype=np.uint8)  # create a layer of the circuit
        mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size]
        if(self.type == "stairwell"):
            rule = f"np.abs(y)<0.1"
            rule = rule.replace('x', f"(mesh[0]/{arr_size})")
            rule = rule.replace('y', f"(mesh[1]/{arr_size})")
            arr *= eval(rule)
        elif(self.type=="dimple"):
            rule = f"np.sqrt(x**2 + y**2) < 0.40 "
            rule = rule.replace('x', f"(mesh[0]/{arr_size})")
            rule = rule.replace('y', f"(mesh[1]/{arr_size})")
            arr *= eval(rule)
        return arr
#marker
    def generate_via_fill(self,z_level,DPI,layer_height):
            if(self.new_step_flag):
                if (self.type == "stairwell"):
                    self.new_step_flag = False
                    arr_size = int(self.width * (DPI / 25.4))
                    arr_height = int(self.step_height / layer_height)
                    current_step = np.floor(((z_level - self.bottom_height) / self.height) * self.step_count)  # decide which stair of the staircase we're on, starting with 0
                    if (current_step % 2 == 0):  # create the function for the stair in the right direction, in normalized coordinates
                        rule = f"(x-0.1)>=-z"
                        mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size, 0:arr_height]

                    else:
                        rule = f"(x-0.1)<=z-1"
                        mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size, arr_height:2*arr_height]
                    #rule = f"np.sqrt(x**2 + y**2) < {0.5 * DPI / 25.4}"
                    arr = np.ones((arr_size, arr_size,arr_height), dtype=np.uint8)
                    rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                    rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                    rule = rule.replace('z', f"(mesh[2]/{2*arr_height})")
                    arr *= eval(rule)
                    return arr
                elif (self.type == "dimple"):
                    self.new_step_flag = False
                    arr_size = int(self.width * (DPI / 25.4))
                    arr_height = int(self.step_height / layer_height)
                    mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size,
                           arr_height:2 * arr_height]
                    rule = f"0"

                    arr = np.ones((arr_size, arr_size,arr_height), dtype=np.uint8)
                    rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                    rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                    rule = rule.replace('z', f"(mesh[2]/{2*arr_height})")
                    arr *= eval(rule)
                    return arr
            return

    def generate_via_fill_layer(self,z_level,DPI,layer_height):
            if(self.new_step_flag):
                if (self.type == "stairwell"):
                    #self.new_step_flag = False
                    arr_size = int(self.width * (DPI / 25.4))
                    arr_height = int(self.step_height / layer_height)
                    current_step = np.floor(((z_level - self.bottom_height) / self.height) * self.step_count)  # decide which stair of the staircase we're on, starting with 0
                    z_normalized = ((z_level - self.bottom_height) - np.floor(
                        current_step / 2) * 2 * self.step_height) / (2 * self.step_height)
                    if (current_step % 2 == 0):  # create the function for the stair in the right direction, in normalized coordinates
                        rule = f"(x-0.1)>=-z"
                        mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size]

                    else:
                        rule = f"(x-0.1)<=z-1"
                        mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size]
                    #rule = f"np.sqrt(x**2 + y**2) < {0.5 * DPI / 25.4}"
                    arr = np.ones((arr_size, arr_size), dtype=np.uint8)
                    rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                    rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                    rule = rule.replace('z', f"{z_normalized}")
                    arr *= eval(rule)
                    return arr
                elif (self.type == "dimple"):
                    #self.new_step_flag = False
                    arr_size = int(self.width * (DPI / 25.4))
                    arr_height = int(self.step_height / layer_height)
                    z_normalized = ((z_level - self.bottom_height) - self.step_height) / (self.step_height)
                    mesh = np.mgrid[-0.5 * arr_size:0.5 * arr_size, -0.5 * arr_size:0.5 * arr_size]
                    rule = f"0"

                    arr = np.ones((arr_size, arr_size), dtype=np.uint8)
                    rule = rule.replace('x', f"(mesh[0]/{arr_size})")
                    rule = rule.replace('y', f"(mesh[1]/{arr_size})")
                    rule = rule.replace('z', f"f{z_normalized}")
                    arr *= eval(rule)
                    return arr
            return

class Circuit_layer():
    def __init__(self,z_height,sliced_model):
        self.sliced_model = sliced_model
        self.resolution = sliced_model.slicing_data["resolution"]
        self.dims = (sliced_model.dims_extended[0],sliced_model.dims_extended[2],sliced_model.dims_extended[1])
        self.offset = sliced_model.offsets
        self.layer_array = np.zeros((self.resolution[0],self.resolution[1]), dtype=np.uint8)
        if(z_height is not None):
            self.z_height = z_height
            self.sandwich_mode = False
        else:
            self.z_height = sliced_model.dims[1][1]-sliced_model.dims[1][0]
            self.sandwich_mode = True
        self.lines = []
        self.pads = []
        self.vias = []
        self.components = []
        self.active_point = None
        self.active_component = None
        self.active_component_angle = 0
        self.line_width = sliced_model.slicing_data["trace_width"]
        self.layer = None
        self.layer_size = (self.dims[0][1]-self.dims[0][0],self.dims[1][1]-self.dims[1][0])
        return

    def draw(self):
        if(self.layer):
            self.layer.draw()
            pass
        return

    def generate_layer(self,render=True,resolution = None):
        self.line_width = self.sliced_model.slicing_data["trace_width"]
        if(not resolution):
            resolution = self.resolution = self.sliced_model.slicing_data["resolution"]
            generate_mode = "edit"
        else:
            generate_mode = "slicer"
        self.layer_array = np.zeros((resolution[0],resolution[1]), dtype=np.uint8)
        for line in self.lines:
            line_scaled = [[0,0],[0,0]]
            line_scaled[0][0] = line[0][0] * resolution[0]
            line_scaled[1][0] = line[1][0] * resolution[0]
            line_scaled[0][1] = line[0][1] * resolution[1]
            line_scaled[1][1] = line[1][1] * resolution[1]
            line_vector = np.array(line_scaled[1])-np.array(line_scaled[0])
            if(line_vector[1]<0): line_vector*=-1

            rules = []
            unit_direction = unit_vector(line_vector)
            rules.append(f" (x-{line_scaled[0][0] + (resolution[0]/self.layer_size[0]) *0.5*self.line_width* unit_direction[1]})*{line_vector[1]} < (y-{line_scaled[0][1] -(resolution[1]/self.layer_size[1]) *0.5*self.line_width* unit_direction[0]}) *{line_vector[0]}")
            rules.append(f" (x-{line_scaled[0][0]-(resolution[0]/self.layer_size[0]) *0.5*self.line_width * unit_direction[1]})*{line_vector[1]} > (y-{line_scaled[0][1] + (resolution[1]/self.layer_size[1]) *0.5*self.line_width * unit_direction[0]}) *{line_vector[0]}")
            rules.append(f"x>{min(line_scaled[0][0],line_scaled[1][0]) - (resolution[0]/self.layer_size[0]) *0.5*self.line_width* unit_direction[1]}")
            rules.append(f"x<{max(line_scaled[0][0], line_scaled[1][0]) +  (resolution[0]/self.layer_size[0]) *0.5*self.line_width* unit_direction[1]}")
            rules.append(f"y>{min(line_scaled[0][1], line_scaled[1][1])- abs( (resolution[0]/self.layer_size[0]) *0.5*self.line_width* unit_direction[0])}")
            rules.append(f"y<{max(line_scaled[0][1], line_scaled[1][1])+ abs( (resolution[0]/self.layer_size[0]) *0.5*self.line_width* unit_direction[0])}")
            self.layer_array = np.logical_or(self.function_layer_array(rules,resolution),self.layer_array)

        for pad in self.pads:
            rules = []
            rules.append(f" x< {np.max((pad[0][0],pad[1][0]))* resolution[0]}")
            rules.append(f" x> -1+ {np.min((pad[0][0],pad[1][0])) * resolution[0]}")
            rules.append(f" y< {np.max((pad[0][1], pad[1][1])) * resolution[1]}")
            rules.append(f" y> -1+ {np.min((pad[0][1], pad[1][1])) * resolution[1]}")
            self.layer_array = np.logical_or(self.function_layer_array(rules, resolution), self.layer_array)

        for component in self.components:
            self.layer_array = np.logical_or(component.generate_pad_layer_array(self.dims,resolution), self.layer_array)


        via_layer_array = np.zeros((resolution[0],resolution[1]), dtype=np.uint8)
        if(generate_mode=="edit"):
            for via in self.vias:
                if(via.origin_layer == self):
                    rules = [f"np.sqrt((x-{via.position[0]*resolution[0]})**2+(y-{via.position[1]*resolution[1]})**2)<{3*(resolution[0]/100)}"]
                else:
                    rules = [f"np.sqrt((x-{via.position[0]*resolution[0]})**2+(y-{via.position[1]*resolution[1]})**2)<{3*(resolution[0]/100)}"]
                via_layer_array = np.logical_or(via_layer_array,self.function_layer_array(rules, resolution))
        self.layer_array = np.logical_and(np.logical_not(via_layer_array),self.layer_array) + (2*via_layer_array).astype(np.uint8)


        if(render):
            #self.layer = Sliced_layer(self.layer_array,self.z_height,0.01,5,[(255,150,150,150)],offset=(self.offset[0],self.offset[2],0),dims=self.dims)
            self.layer = Sliced_layer(self.layer_array, self.z_height, 0.01, 5, [self.sliced_model.slicing_data["silver_color"],(255, 150, 150,255)], dims=self.dims) #render_edit
        return

    def function_layer_array(self, rules,resolution):
        arr = np.ones((resolution[0], resolution[1]), dtype=np.uint8)
        mesh = np.mgrid[0:resolution[0],0:resolution[1]]
        for rule in rules:
            rule = rule.replace('x', "mesh[0]")
            rule = rule.replace('y', "mesh[1]")
            arr *= eval(rule)
        return arr

    def add_line(self,coords):
        layer_coords = self.get_layer_coords(coords)
        if(layer_coords[0]<0 or layer_coords[1]<0 or layer_coords[0]>1 or layer_coords[1] > 1):
            return
        clamping_distance = 0.02
        for line in self.lines:
            for point in line:
                if( abs(point[0] - layer_coords[0]) < clamping_distance and abs(point[1] - layer_coords[1]) < clamping_distance ):
                    layer_coords[0] = point[0]
                    layer_coords[1] = point[1]

        if(not self.active_point):
            self.active_point = layer_coords
            return
        else:
            self.lines.append( (self.active_point, layer_coords) )
            self.active_point = None
            global tracer_help_shape
            tracer_help_shape = None
        self.generate_layer()
        return

    def remove_via(self,coords):
        if(self.active_point):
            self.active_point = None
            return
        layer_coords = self.get_layer_coords(coords)
        clamping_distance = 0.05
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return
        for via in model.circuit.vias:
            #print(np.sqrt((via.position[0] - layer_coords[0])**2 +(via.position[1] - layer_coords[1])**2 ))
            if(np.sqrt((via.position[0] - layer_coords[0])**2 +(via.position[1] - layer_coords[1])**2 ) <  clamping_distance):
                via.origin_layer.vias.remove(via)
                via.exit_layer.vias.remove(via)
                model.circuit.vias.remove(via)
        self.generate_layer()
        return

    def remove_line(self,coords):
        if(self.active_point):
            self.active_point = None
            global tracer_help_shape
            tracer_help_shape = None
            return
        layer_coords = self.get_layer_coords(coords)
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return
        clamping_distance = 0.03
        for line in self.lines[:]:
            line_vector = np.array(line[1]) - np.array(line[0])
            perp_line_vector = np.array([line_vector[1],-line_vector[0]])
            dist_line =  abs(perp_line_vector[0]*(layer_coords[0]-line[0][0]) + perp_line_vector[1]*(layer_coords[1]-line[0][1]) )/np.sqrt(line_vector[0]**2 + line_vector[1]**2)
            if(   (layer_coords[0]-line[0][0])/line_vector[0] <0 or (layer_coords[0]-line[0][0])/line_vector[0] > 1):
                dist_line = 10
            dist_circle_0 = np.sqrt((layer_coords[0]-line[0][0])**2 + (layer_coords[1]-line[0][1])**2)
            dist_circle_1 = np.sqrt((layer_coords[1] - line[1][0]) ** 2 + (layer_coords[1] - line[1][1]) ** 2)
            min_dist = min(dist_line,dist_circle_0,dist_circle_1)
            if(min_dist<clamping_distance):
                self.lines.remove(line)
                self.generate_layer()
                return

        return

    def add_pad(self,coords):
        layer_coords = self.get_layer_coords(coords)
        if(layer_coords[0]<0 or layer_coords[1]<0 or layer_coords[0]>1 or layer_coords[1] > 1):
            return

        if(not self.active_point):
            self.active_point = layer_coords
            return
        else:
            self.pads.append( (self.active_point, layer_coords) )
            self.active_point = None
            global tracer_help_shape
            tracer_help_shape = None
        self.generate_layer()
        return

    def remove_pad(self,coords):
        if(self.active_point):
            self.active_point = None
            global tracer_help_shape
            tracer_help_shape = None
            return
        layer_coords = self.get_layer_coords(coords)
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return
        for pad in self.pads:
            if(layer_coords[0] > np.min((pad[0][0],pad[1][0])) and layer_coords[0] < np.max((pad[0][0],pad[1][0]))):
                if (layer_coords[1] > np.min((pad[0][1], pad[1][1])) and layer_coords[1] < np.max((pad[0][1], pad[1][1]))):
                    self.pads.remove(pad)
        self.generate_layer()
        return

    def add_via(self,coords):
        layer_coords = self.get_layer_coords(coords)
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return
        clamping_distance = 0.02
        for line in self.lines:
            for point in line:
                if (abs(point[0] - layer_coords[0]) < clamping_distance and abs(
                        point[1] - layer_coords[1]) < clamping_distance):
                    layer_coords[0] = point[0]
                    layer_coords[1] = point[1]
        base = Tk()
        field_spacing = 40
        field_y0 = 20
        base.geometry(f"500x{(len(model.circuit.circuit_layers) + 2) * field_spacing}")
        base.title("Connect to circuit layer")

        for i, layer in enumerate(model.circuit.circuit_layers):
            if(layer != self):
                Button(base, text=f"z={layer.z_height:.2f}", width=20,command=partial(self.create_via,layer,base,layer_coords)).place(x=100,y=i * field_spacing + field_y0)
        Button(base, text="Cancel", width=10, command=lambda: base.destroy()).place(x=10, y=(len(model.circuit.circuit_layers)) * field_spacing + field_y0)
        base.mainloop()
        return

    def create_via(self,circuit_layer,base,layer_coords):
        base.destroy()
        via = Via(self,circuit_layer,layer_coords,via_mode)
        self.vias.append(via)
        circuit_layer.vias.append(via)
        model.circuit.vias.append(via)
        self.generate_layer()
        return


    def add_component(self,coords):
        layer_coords = self.get_layer_coords(coords)
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return

        angle = self.active_component_angle
        self.components.append(self.active_component(layer_coords, angle, self.z_height))
        self.generate_layer()
        return

    def remove_component(self,coords):
        if(self.active_component):
            self.active_component = None
            global tracer_preview_img
            tracer_preview_img = None
            return
        layer_coords = self.get_layer_coords(coords)
        clamping_distance = 0.05
        if (layer_coords[0] < 0 or layer_coords[1] < 0 or layer_coords[0] > 1 or layer_coords[1] > 1):
            return
        for component in self.components:
            #print(np.sqrt((via.position[0] - layer_coords[0])**2 +(via.position[1] - layer_coords[1])**2 ))
            if(np.sqrt((component.position[0] - layer_coords[0])**2 +(component.position[1] - layer_coords[1])**2 ) <  clamping_distance):
                self.components.remove(component)
        self.generate_layer()
        return



    def get_layer_coords(self,coords):

        layer_coords = np.array(coords[0,0:3]).flatten()
        layer_coords[0] -= self.dims[0][0] + self.offset[0]
        layer_coords[0] /= (self.dims[0][1] - self.dims[0][0])

        layer_coords[2] -= self.dims[1][0] + self.offset[2]
        layer_coords[2] /= (self.dims[1][1] - self.dims[1][0])
        return [layer_coords[0] ,1-layer_coords[2]]
    
    def get_global_coords(self,coords):
        global_coords = np.asarray([coords[0],1-coords[1]])

        global_coords[1] *= (self.dims[1][1] - self.dims[1][0])
        global_coords[1] += self.dims[1][0] + self.offset[2]

        global_coords[0] *= (self.dims[0][1] - self.dims[0][0])
        global_coords[0] += self.dims[0][0] + self.offset[0]
        return global_coords


class Circuit():
    def __init__(self):
        self.circuit_layers = []
        self.vias = []

        self.component_list = [components.Component_smd_0805_buried,components.Component_smd_0805_non_buried,components.Component_ATtiny85,components.Component_pad_4x4_open]
    def create_circuit_layer(self,editor=True):
        if(editor):
            global program_state
            program_state = "tracer"
            window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR))
            camera.fix(-0.5 * np.radians(180), 0, 0)
        z_level = model.activelayer_height

        global active_circuit_layer
        for circuit_layer in self.circuit_layers:
            if(np.abs(circuit_layer.z_height - z_level)<0.5*model.slicing_data["layer_height"]):
                active_circuit_layer = circuit_layer
                active_circuit_layer.generate_layer(render=True)
                return
        circuit_layer = Circuit_layer(z_level,model)
        self.circuit_layers.append(circuit_layer)
        active_circuit_layer = circuit_layer

    def create_top_circuit_layer(self,editor=True):
        if (editor):
            global program_state
            program_state = "tracer"
            window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR))
            camera.fix(-0.5 * np.radians(180), 0, 0)
        z_level = model.activelayer_height
        # for circuit_layer in self.circuit_layers:
        #     print(np.abs(circuit_layer.z_height - z_level))
        #     if (np.abs(circuit_layer.z_height - z_level) < 0.5 * model.slicing_data["layer_height"]):
        #         global active_circuit_layer
        #         active_circuit_layer = circuit_layer
        #         return
        global active_circuit_layer
        circuit_layer = Circuit_layer(None, model)
        self.circuit_layers.append(circuit_layer)
        active_circuit_layer = circuit_layer

def unit_vector(v):
    return v / np.linalg.norm(v)

config = pg.gl.Config(sample_buffers=1, samples=4, depth_size=16, double_buffer=True,alpha_size=8)
window = pg.window.Window(width=960, height=540, resizable=True, config=config)
camera = cam.Camera()
keys = pg.window.key.KeyStateHandler()
window.push_handlers(keys)
window.set_caption("SquidSlicer V1.0")
slicer_icon = pg.resource.image("ui_icons/octo.ico")
window.set_icon(slicer_icon)

@window.event
def on_draw():
    global model
    global ui_slicer_batch
    global ui_tracer_batch
    global plate_batch
    global program_state
    window.view = camera.get_view_matrix().getA().flatten()
    #mono_shader_program['pureRotation'] = np.linalg.inv(camera.rot_matrix).getA().flatten()

    window.clear()

    plate_batch.draw()
    model.draw(camera)
    if (camera.angles[0] > 0):
        plate_batch.draw()


    width,height = window.get_framebuffer_size()

    global active_circuit_layer
    if(program_state == "tracer"):
        texture_shader_program['offset'] = pg.math.Mat4.from_translation(
            pg.math.Vec3(model.offsets[0], model.offsets[1],
                         model.offsets[2]))
        active_circuit_layer.draw()
        texture_shader_program['offset'] = pg.math.Mat4()

        pass
#    else:
#        for circuit_layer in circuit_layers:
            #circuit_layer.draw()
#            pass

    window.projection = pg.math.Mat4.orthogonal_projection(1, z_near=-1, z_far=512, right=width, bottom=-0, top=height)
    window.view = np.eye(4).flatten()
    if(program_state=="slicer"):
        ui_slicer_batch.draw()
    if (program_state == "tracer"):
        ui_tracer_batch.draw()

    window.projection = pg.math.Mat4.orthogonal_projection(window.aspect_ratio, z_near=0.01, z_far=512,
                                                           right=-window.aspect_ratio,
                                                           bottom=-1, top=1)

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global program_state
    if (keys[pg.window.key.LCTRL]):
        camera.zoom(10 ** (scroll_y / 50))
    else:
        if((program_state=="slicer" and model.render_mode=="sliced") or program_state=="tracer"):
            if (scroll_y < 0):
                model.activelayer_height -= model.slicing_data["layer_height"]
            elif (scroll_y > 0):
                model.activelayer_height += model.slicing_data["layer_height"]
            model.activelayer_height = max(0,model.activelayer_height)
            model.activelayer_height = min(model.activelayer_height,model.layers[-1].z_level +
                                           model.slicing_data["layer_height"])

            for layer in model.layers:
                if(model.activelayer_height >= layer.print_level):
                   layer.visible =True
                else:
                   layer.visible = False
    pass


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if(buttons==1):
        camera.rotate(-dy/100,-dx/100,0)
    elif(buttons==2):
        camera.translate(dx,dy,0)
    return

@window.event
def on_resize(width, height):
    window.viewport = (0, 0, width, height)
    print(window.size)
    #window.projection = pg.math.Mat4.perspective_projection(window.aspect_ratio, z_near=0.01, z_far=255, fov=60)
    #window.projection = pg.math.Mat4.orthogonal_projection(window.aspect_ratio, z_near=0.01, z_far=512,right=-window.aspect_ratio,bottom=-1,top=1)
    return pg.event.EVENT_HANDLED



@window.event
def on_mouse_press(x, y, button, modifiers):
    global program_state
    if(program_state=="tracer" and not (y<80 and x<480) ):
        vec = np.array([x,y,0,1])
        view_matrix = camera.get_view_matrix()
        projection_matrix_3d = np.reshape(np.asarray(window.projection),(4,4))
        projection_matrix_2d = np.reshape(np.asarray(pg.math.Mat4.orthogonal_projection(1, z_near=-1, z_far=512, right=window.width, bottom=-0, top=window.height)),(4,4))
        model_coord = ((vec @ projection_matrix_2d) @ np.linalg.inv(projection_matrix_3d)) @ np.linalg.inv(view_matrix)
        #model_coord[0, 2] *= -1
        if(tracer_state == "line"):
            if(button==1):
                active_circuit_layer.add_line(model_coord)
            if(button ==4):
                active_circuit_layer.remove_line(model_coord)
        if(tracer_state == "via"):
            if (button == 1):
                active_circuit_layer.add_via(model_coord)
            if (button == 4):
                active_circuit_layer.remove_via(model_coord)
        if(tracer_state == "pad"):
            if (button == 1):
                active_circuit_layer.add_pad(model_coord)
            if (button == 4):
                active_circuit_layer.remove_pad(model_coord)
        if(tracer_state== "component"):
            if (button == 1):
                active_circuit_layer.add_component(model_coord)
            if (button == 4):
                active_circuit_layer.remove_component(model_coord)
    pass

@window.event
def on_mouse_motion(x,y,dx,dy):
    if (program_state == "tracer" and not (y < 80 and x < 480)):
        vec = np.array([x, y, 0, 1])
        view_matrix = camera.get_view_matrix()
        projection_matrix_3d = np.reshape(np.asarray(window.projection), (4, 4))
        projection_matrix_2d = np.reshape(np.asarray(
            pg.math.Mat4.orthogonal_projection(1, z_near=-1, z_far=512, right=window.width, bottom=-0,
                                               top=window.height)), (4, 4))
        model_coord = ((vec @ projection_matrix_2d) @ np.linalg.inv(projection_matrix_3d)) @ np.linalg.inv(view_matrix)
        global ui_tracer_batch
        global tracer_help_shape
        global active_circuit_layer
        if (tracer_state == "line"):
            if(active_circuit_layer.active_point is not None):
                active_point_x_global,active_point_y_global = active_circuit_layer.get_global_coords(active_circuit_layer.active_point)
                vec_global_coords = np.array([active_point_x_global, 0, active_point_y_global, 1])
                screen_coord = (((vec_global_coords @ view_matrix) @ projection_matrix_3d) @ np.linalg.inv(projection_matrix_2d))

                layer_coord = active_circuit_layer.get_layer_coords(model_coord)
                if(layer_coord[0] > 0 and layer_coord[0] < 1 and layer_coord[1] > 0 and layer_coord[1] < 1 ):
                    color = (155, 214, 92,180)
                else:
                    color = (214, 104, 92,180)

                tracer_help_shape = pg.shapes.Line(screen_coord[0,0],screen_coord[0,1],x,y,5,color=color,batch = ui_tracer_batch)
        if(tracer_state=="pad"):
            if (active_circuit_layer.active_point is not None):
                active_point_x_global, active_point_y_global = active_circuit_layer.get_global_coords(
                    active_circuit_layer.active_point)
                vec_global_coords = np.array([active_point_x_global, 0, active_point_y_global, 1])
                screen_coord = (((vec_global_coords @ view_matrix) @ projection_matrix_3d) @ np.linalg.inv(
                    projection_matrix_2d))

                layer_coord = active_circuit_layer.get_layer_coords(model_coord)
                if (layer_coord[0] > 0 and layer_coord[0] < 1 and layer_coord[1] > 0 and layer_coord[1] < 1):
                    color = (155, 214, 92,180)
                else:
                    color = (214, 104, 92,180)

                tracer_help_shape = pg.shapes.Rectangle(screen_coord[0, 0], screen_coord[0, 1], x-screen_coord[0, 0], y-screen_coord[0, 1], color=color,
                                                  batch=ui_tracer_batch)
        if (tracer_state == "component"):
            global activate_circuit_layer
            if(active_circuit_layer.active_component):
                tracer_preview_img.x = x + tracer_preview_img.width*0.5
                tracer_preview_img.y = y - tracer_preview_img.height*0.5
                tracer_preview_img.scale = 0.5*(window.height/ tracer_preview_img.image.height) * (model.dims[2][1]-model.dims[2][0]) * camera.zoom_matrix[0,0]

    pass

def load_plate_model():
    global plate_batch
    plate_pic = pg.image.load('plate2.png')
    plate_tex = plate_pic.get_texture()
    plate_batch = pg.graphics.Batch()
    plate_RenderGroup = RenderGroupTextured(plate_tex, texture_shader_program,True)
    vertices = [100,0,100,-100,0,100,100,0,-100,-100,0,-100]
    #vertices = [50, 0, 50, -50, 0, 50, 50, 0, -50, -50, 0, -50] #render edit
    indices = [1, 2, 0, 1, 3, 2]
    vertex_list = texture_shader_program.vertex_list_indexed(len(vertices) // 3, pg.gl.GL_TRIANGLES, indices, plate_batch, plate_RenderGroup,
                                             position=('f', vertices),
                                             tex_coords=('f', (0, 0, 0,   1, 0, 0,   0, 1, 0,   1, 1, 0)))
    return pg.model.Model([vertex_list], [plate_RenderGroup], plate_batch)

def add_stl_model():
    global program_state
    if(program_state=="slicer"):
        root = Tk()
        root.withdraw()
        file_names = askopenfilenames(filetypes=[("3d model", ".stl")])
        root.destroy()
        if(file_names):
            if(not model.filename):
                model.load_stl(file_names[0])
                for file_name in file_names[1:]:
                    model.add_secondary_model(file_name)
            else:
                for file_name in file_names:
                    model.add_secondary_model(file_name)
                pass

def render_slice():
    global program_state
    if(program_state=="slicer"):
        if(model.filename):
            #global settings
            model.slicing_data["DPI"] = int(menus_setup.settings["render_DPI"]["value"])
            model.slicing_data["layer_height"] = float(menus_setup.settings["render_layer_height"]["value"])
            model.slicing_data["support_generation"] = bool(menus_setup.settings["support_generation"]["value"])
            model.slicing_data["render_components"] = bool(menus_setup.settings["render_components"]["value"])
            model.slicing_data["slice_traces"] = bool(menus_setup.settings["traces_slicing"]["value"])
            opacity = max(1,min(int(400*model.slicing_data["layer_height"]),255))
            #model.slicing_data["base_color"] = (opacity, 250, 200, 0)
            model.slicing_data["support_spacing"] = int(menus_setup.settings["support_spacing"]["value"])
            model.slicing_data["traces_print_height"] = float(menus_setup.settings["traces_print_height"]["value"])
            model.slicing_data["trace_width"] = float(menus_setup.settings["trace_width"]["value"])
            model.slicing_data["secondary_color"] = (model.slicing_data["secondary_color"][0],
            model.slicing_data["secondary_color"][1], model.slicing_data["secondary_color"][2],int(menus_setup.settings["sec_struc_transparancy"]["value"]))

           # model.slicing_data["base_color"] = (model.slicing_data["base_color"][0],
            #model.slicing_data["base_color"][1], model.slicing_data["base_color"][2],int(menus_setup.settings["sec_struc_transparancy"]["value"])) #render edit

            model.slice(export=False)
        return

def export_slice():
    global program_state
    if(program_state=="slicer"):
        if (model.visible):
            root = Tk()
            root.withdraw()
            file_name = asksaveasfilename(filetypes=[("bitmap stack", ".png")])

            root.destroy()
            print(f"Open file: {file_name}")
            if(file_name):
                model.slicing_data["DPI"] = int(menus_setup.settings["slice_DPI"]["value"])
                model.slicing_data["layer_height"] = float(menus_setup.settings["slice_layer_height"]["value"])
                model.slicing_data["support_generation"] = bool(menus_setup.settings["support_generation"]["value"])
                model.slicing_data["slice_traces"] = bool(menus_setup.settings["traces_slicing"]["value"])
                model.slicing_data["support_spacing"] = int(menus_setup.settings["support_spacing"]["value"])
                model.slicing_data["traces_print_height"] = float(menus_setup.settings["traces_print_height"]["value"])
                model.slicing_data["trace_width"] = float(menus_setup.settings["trace_width"]["value"])
                model.slicing_data["printhead_UV_offset"] = float(menus_setup.settings["printhead_UV_offset"]["value"])
                os.makedirs(file_name)
                model.slice(export=file_name)
        return

def center():
    global program_state
    global model
    if(program_state=="slicer"):
        if(model.filename):
            dims = model.dims_extended
            model.offsets = (-dims[0][0]-0.5*(dims[0][1]-dims[0][0]), -dims[1][0], -dims[2][0]-0.5*(dims[2][1]-dims[2][0]) )
            for circuit_layer in model.circuit.circuit_layers:
                circuit_layer.offset = model.offsets
        return

def setup():
    pg.gl.glClearColor(1, 1, 1, 1)
    on_resize(*window.size)
    load_plate_model()
    global model
    model = Object()
    global active_circuit_layer
    active_circuit_layer = None
    texture_shader_program['offset'] =pg.math.Mat4.from_translation(pg.math.Vec3(0, 0,0))
    mono_shader_program['offset'] = pg.math.Mat4.from_translation(pg.math.Vec3(0, 0, 0))
    #mono_shader_program['pureRotation'] =  pg.math.Mat4.from_translation(pg.math.Vec3(0, 0, 0))
    global program_state
    global tracer_state
    global via_mode
    global tracer_help_shape
    global tracer_preview_img
    program_state = "slicer"
    tracer_state = "line"
    via_mode = "stairwell"
    tracer_help_shape = None
    tracer_preview_img = None



def toggle_tracer():
    if(program_state=="slicer" and model.render_mode=="sliced"):
        base = Tk()
        base.geometry(f"300x100")
        base.title("Create circuit layer")
        Button(base, text=f"Create at current layer", width=20, command=lambda: (base.destroy(),model.circuit.create_circuit_layer())).place(x=10,y=10)
        Button(base, text=f"Create on main structure", width=20, command=lambda: (base.destroy(),model.circuit.create_top_circuit_layer())).place(x=10, y=40)
        Button(base, text="Cancel", width=10, command=lambda: base.destroy()).place(x=10,y=70)
        base.mainloop()
    return pg.event.EVENT_HANDLED


def toggle_slicer():
    global program_state
    if (program_state == "tracer"):
        program_state = "slicer"
        camera.unfix()
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_DEFAULT))
        return

def settings_menu():
    if (program_state == "slicer"):
        menus_setup.settings_menu()
    return

def tracer_menu():
    global active_circuit_layer
    global model
    global program_state
    if (program_state == "slicer"):
        model.circuit.circuit_layers,program_state,active_circuit_layer, = menus_setup.tracer_menu(model.circuit.circuit_layers,active_circuit_layer)
    if(program_state == "tracer"):
        window.set_mouse_cursor(window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR))
        camera.fix(-0.5 * np.radians(180), 0, 0)
        model.activelayer_height = active_circuit_layer.z_height
        active_circuit_layer.generate_layer(render=True)
        for layer in model.layers:
            if (model.activelayer_height >= layer.print_level):
                layer.visible = True
            else:
                layer.visible = False
    return

def add_line():
    global tracer_state
    if (program_state == "tracer"):
        tracer_state = "line"
    return

def add_stairwell_via():
    global tracer_state
    global via_mode
    if (program_state == "tracer"):
        tracer_state = "via"
        via_mode = "stairwell"
    return

def add_component():
    if (program_state == "tracer"):
        global tracer_state
        base = Tk()
        field_spacing = 40
        field_y0 = 20
        base.geometry(f"500x{(len(model.circuit.component_list) + 3) * field_spacing}")
        base.title("Add electronic component")
        Button(base, text="Cancel", width=10, command=lambda: base.destroy()).place(x=10, y=(
                                                                                                    len(model.circuit.component_list) + 1) * field_spacing + field_y0)
        Label(base, text="Enter component angle:",
              font=("arial", 12)).place(x=10, y=(len(model.circuit.component_list)) * field_spacing + field_y0)
        enter_angle = Entry(base)
        enter_angle.place(x=200, y=(len(model.circuit.component_list)) * field_spacing + field_y0)
        enter_angle.insert(0, 0)

        for i, component in enumerate(model.circuit.component_list):
            Button(base, text=f"{component.display_name()}", width=40,
                   command=partial(activate_component, component, base, enter_angle)).place(x=10,
                                                                                                             y=i * field_spacing + field_y0)
        base.mainloop()


def activate_component(component,base,angle):
    global active_circuit_layer
    global tracer_state
    global tracer_preview_img
    active_circuit_layer.active_component_angle = 2 * np.pi * float(angle.get()) / 360
    base.destroy()
    active_circuit_layer.active_component = component
    tracer_state = "component"
    dummy_component = component((0.5,0.5),active_circuit_layer.active_component_angle,0)
    dummy_component_pad_layer_array = dummy_component.generate_pad_layer_array(active_circuit_layer.dims,active_circuit_layer.resolution)
    dummy_component_pad_layer = Sliced_layer(dummy_component_pad_layer_array.astype(np.uint8),0,1,1,[(92,214,155,180)],dims=active_circuit_layer.dims)
    tracer_preview_img = pg.sprite.Sprite(dummy_component_pad_layer.tex.get_transform(flip_x=True), 0, 0, batch=ui_tracer_batch)


def add_dimple_via():
    global tracer_state
    global via_mode
    if (program_state == "tracer"):
        tracer_state = "via"
        via_mode = "dimple"
    return

def add_pad():
    global tracer_state
    if (program_state == "tracer"):
        tracer_state = "pad"
    return

def clear_model():
    global program_state
    if (program_state == "slicer"):
        model.clear()
    return

textured_shader_v,textured_shader_f = shaders_textured.define_shader()
texture_shader_program = pg.graphics.shader.ShaderProgram(pg.graphics.shader.Shader(textured_shader_v, 'vertex'), pg.graphics.shader.Shader(textured_shader_f, 'fragment'))
mono_shader_v,mono_shader_f = shaders_mono.define_shader()
mono_shader_program = pg.graphics.shader.ShaderProgram(pg.graphics.shader.Shader(mono_shader_v, 'vertex'), pg.graphics.shader.Shader(mono_shader_f, 'fragment'))


ui_slicer_batch = pg.graphics.Batch()
ui_tracer_batch = pg.graphics.Batch()

toggle_mode_button2 = ui_setup.create_button(pg,window,"ui_icons/model.png",ui_tracer_batch,toggle_slicer)
add_stairwell_via_button = ui_setup.create_button(pg,window,"ui_icons/add_stairwell_via.png",ui_tracer_batch,add_stairwell_via)
add_dimple_via_button = ui_setup.create_button(pg,window,"ui_icons/add_dimple_via.png",ui_tracer_batch,add_dimple_via)
add_pad_button = ui_setup.create_button(pg,window,"ui_icons/add_pad.png",ui_tracer_batch,add_pad)
add_component_button = ui_setup.create_button(pg,window,"ui_icons/add_component.png",ui_tracer_batch,add_component)
add_line_button = ui_setup.create_button(pg,window,"ui_icons/add_line.png",ui_tracer_batch,add_line)

plus_button = ui_setup.create_button(pg,window,"ui_icons/plus.png",ui_slicer_batch,add_stl_model,True)
center_button = ui_setup.create_button(pg,window,"ui_icons/center.png",ui_slicer_batch,center)
render_slice_button = ui_setup.create_button(pg,window,"ui_icons/supports.png",ui_slicer_batch,render_slice)
export_slice_button = ui_setup.create_button(pg,window,"ui_icons/print.png",ui_slicer_batch,export_slice)
settings_menu_button = ui_setup.create_button(pg,window,"ui_icons/settings.png",ui_slicer_batch,settings_menu)
toggle_mode_button = ui_setup.create_button(pg,window,"ui_icons/traces.png",ui_slicer_batch,toggle_tracer)
tracer_menu_button = ui_setup.create_button(pg,window,"ui_icons/traces_settings.png",ui_slicer_batch,tracer_menu)
trash_button = ui_setup.create_button(pg,window,"ui_icons/trash_can.png",ui_slicer_batch,clear_model)


setup()
pg.app.run()