import numpy as np
import stl
np.seterr(divide='ignore', invalid='ignore') # I use the divide by zero and nan so ignore the warnings

def slice_model_3d(filename,resolution_x,layer_height):
    mesh = stl.mesh.Mesh.from_file(filename)
    mesh.rotate([0.5,0,0],np.radians(-90))
    model_max_x = np.max((np.max(mesh.v0[:,0]),np.max(mesh.v1[:,0]),np.max(mesh.v2[:,0])))
    model_min_x = np.min((np.min(mesh.v0[:,0]),np.min(mesh.v1[:,0]),np.min(mesh.v2[:,0])))
    model_size_x = model_max_x - model_min_x
    mm_per_pixel = model_size_x / resolution_x

    model_max_y = np.max((np.max(mesh.v0[:,1]),np.max(mesh.v1[:,1]),np.max(mesh.v2[:,1])))
    model_min_y = np.min((np.min(mesh.v0[:,1]),np.min(mesh.v1[:,1]),np.min(mesh.v2[:,1])))
    model_size_y = model_max_y - model_min_y
    resolution_y = int(np.ceil(model_size_y / mm_per_pixel))

    model_max_z = np.max((np.max(mesh.v0[:, 2]), np.max(mesh.v1[:, 2]), np.max(mesh.v2[:, 2])))
    model_min_z = np.min((np.min(mesh.v0[:, 2]), np.min(mesh.v1[:, 2]), np.min(mesh.v2[:, 2])))
    model_size_z = model_max_z - model_min_z

    resolution_z = int(model_size_z/layer_height)
    layer_array = np.zeros((resolution_x,resolution_y,resolution_z), dtype = np.uint8)

    for layer in range(0,resolution_z):
        layer_z = layer * layer_height + model_min_z
        print(f"processing layer {layer_z}")
        intersects_layer = np.logical_not( (mesh.v0[:,2]>layer_z)*(mesh.v1[:,2]>layer_z)*(mesh.v2[:,2]>layer_z)+(mesh.v0[:,2]<layer_z)*(mesh.v1[:,2]<layer_z)*(mesh.v2[:,2]<layer_z) )

        intersected_elements_indices = np.nonzero(intersects_layer)[0]

        intersected_v0 = mesh.v0[intersected_elements_indices,:]
        intersected_v1 = mesh.v1[intersected_elements_indices,:]
        intersected_v2 = mesh.v2[intersected_elements_indices,:]

        vector1 = intersected_v1-intersected_v0
        vector2 = intersected_v2-intersected_v1
        vector3 = intersected_v0-intersected_v2

        vector1_coeff = np.atleast_2d(((layer_z - intersected_v0[:,2]) / vector1[:,2])).T
        vector2_coeff = np.atleast_2d(((layer_z - intersected_v1[:,2]) / vector2[:,2])).T
        vector3_coeff = np.atleast_2d(((layer_z - intersected_v2[:,2]) / vector3[:,2])).T

        vector1_intersect = np.logical_and(vector1_coeff>0,vector1_coeff<1)
        vector2_intersect = np.logical_and(vector2_coeff>0,vector2_coeff<1)
        vector3_intersect = np.logical_and(vector3_coeff>0,vector3_coeff<1)

        vector1_non_intersect_indices = np.nonzero(np.logical_not(vector1_intersect[:,0]))[0]
        vector2_non_intersect_indices = np.nonzero(np.logical_not(vector2_intersect[:,0]))[0]
        vector3_non_intersect_indices = np.nonzero(np.logical_not(vector3_intersect[:,0]))[0]

        intersect_points_1 =  vector1_coeff * vector1 + intersected_v0
        intersect_points_2 =  vector2_coeff * vector2 + intersected_v1
        intersect_points_3 =  vector3_coeff * vector3 + intersected_v2

        intersect_points_1[vector1_non_intersect_indices,:] = np.nan
        intersect_points_2[vector2_non_intersect_indices,:] = np.nan
        intersect_points_3[vector3_non_intersect_indices,:] = np.nan
        intersect_points_1 = intersect_points_1[:,0:2]
        intersect_points_2 =  intersect_points_2[:,0:2]
        intersect_points_3 =  intersect_points_3[:,0:2]

        intersect_points = np.concatenate((intersect_points_1,intersect_points_2,intersect_points_3),axis=1)
        intersect_points = intersect_points[np.isfinite(intersect_points)]
        intersect_points = np.reshape(intersect_points,(int(len(intersect_points)/4),4))


        for x in range(0,resolution_x):
            line_x = model_min_x + x * mm_per_pixel + 0.5 * mm_per_pixel

            intersects_line = np.logical_xor((intersect_points[:,0]>line_x),(intersect_points[:,2]>line_x))


            intersected_lines_indices = np.nonzero(intersects_line)[0]
            intersected_lines = intersect_points[intersected_lines_indices,:]

            layer_vectors = intersected_lines[:,2:4]-intersected_lines[:,0:2]
            layer_vector_origins = intersected_lines[:,0:2]
            intersect_points_1d = np.atleast_2d((line_x-layer_vector_origins[:,0])/layer_vectors[:,0]).T * layer_vectors + layer_vector_origins
            edge_pixels = np.round( (intersect_points_1d[:,1]-model_min_y) / mm_per_pixel  )
            edge_pixels = np.sort(edge_pixels)
            for i in range(0,int(len(edge_pixels)/2)):
                layer_array[ x,int(edge_pixels[i*2]) : int((edge_pixels[i*2+1])),layer] = 1#layer_z
    return layer_array, ((model_min_x,model_max_x),(model_min_y,model_max_y),(model_min_z,model_max_z))

def get_model_dims(mesh):
    model_max_x = np.max((np.max(mesh.v0[:, 0]), np.max(mesh.v1[:, 0]), np.max(mesh.v2[:, 0])))
    model_min_x = np.min((np.min(mesh.v0[:, 0]), np.min(mesh.v1[:, 0]), np.min(mesh.v2[:, 0])))
    model_size_x = model_max_x - model_min_x

    model_max_y = np.max((np.max(mesh.v0[:, 1]), np.max(mesh.v1[:, 1]), np.max(mesh.v2[:, 1])))
    model_min_y = np.min((np.min(mesh.v0[:, 1]), np.min(mesh.v1[:, 1]), np.min(mesh.v2[:, 1])))
    model_size_y = model_max_y - model_min_y

    model_max_z = np.max((np.max(mesh.v0[:, 2]), np.max(mesh.v1[:, 2]), np.max(mesh.v2[:, 2])))
    model_min_z = np.min((np.min(mesh.v0[:, 2]), np.min(mesh.v1[:, 2]), np.min(mesh.v2[:, 2])))
    model_size_z = model_max_z - model_min_z
    return ((model_min_x,model_max_x),(model_min_y,model_max_y),(model_min_z,model_max_z))


def slice_model_2d(mesh,layer_z,DPI):
    model_max_x = np.max((np.max(mesh.v0[:,0]),np.max(mesh.v1[:,0]),np.max(mesh.v2[:,0])))
    model_min_x = np.min((np.min(mesh.v0[:,0]),np.min(mesh.v1[:,0]),np.min(mesh.v2[:,0])))
    model_size_x = model_max_x - model_min_x
    resolution_x = int(round((model_size_x/25.4)*DPI))
    mm_per_pixel = model_size_x / resolution_x

    model_max_y = np.max((np.max(mesh.v0[:,1]),np.max(mesh.v1[:,1]),np.max(mesh.v2[:,1])))
    model_min_y = np.min((np.min(mesh.v0[:,1]),np.min(mesh.v1[:,1]),np.min(mesh.v2[:,1])))
    model_size_y = model_max_y - model_min_y
    resolution_y = int(round((model_size_y/25.4)*DPI))

    model_max_z = np.max((np.max(mesh.v0[:, 2]), np.max(mesh.v1[:, 2]), np.max(mesh.v2[:, 2])))
    model_min_z = np.min((np.min(mesh.v0[:, 2]), np.min(mesh.v1[:, 2]), np.min(mesh.v2[:, 2])))
    model_size_z = model_max_z - model_min_z
    layer_array = np.zeros((resolution_x, resolution_y), dtype=np.uint8)

    #print(f"processing layer {layer_z}")
    intersects_layer = np.logical_not( (mesh.v0[:,2]>layer_z)*(mesh.v1[:,2]>layer_z)*(mesh.v2[:,2]>layer_z)+(mesh.v0[:,2]<layer_z)*(mesh.v1[:,2]<layer_z)*(mesh.v2[:,2]<layer_z) )

    intersected_elements_indices = np.nonzero(intersects_layer)[0]

    intersected_v0 = mesh.v0[intersected_elements_indices,:]
    intersected_v1 = mesh.v1[intersected_elements_indices,:]
    intersected_v2 = mesh.v2[intersected_elements_indices,:]

    vector1 = intersected_v1-intersected_v0
    vector2 = intersected_v2-intersected_v1
    vector3 = intersected_v0-intersected_v2

    vector1_coeff = np.atleast_2d(((layer_z - intersected_v0[:,2]) / vector1[:,2])).T
    vector2_coeff = np.atleast_2d(((layer_z - intersected_v1[:,2]) / vector2[:,2])).T
    vector3_coeff = np.atleast_2d(((layer_z - intersected_v2[:,2]) / vector3[:,2])).T

    vector1_intersect = np.logical_and(vector1_coeff>0,vector1_coeff<1)
    vector2_intersect = np.logical_and(vector2_coeff>0,vector2_coeff<1)
    vector3_intersect = np.logical_and(vector3_coeff>0,vector3_coeff<1)

    vector1_non_intersect_indices = np.nonzero(np.logical_not(vector1_intersect[:,0]))[0]
    vector2_non_intersect_indices = np.nonzero(np.logical_not(vector2_intersect[:,0]))[0]
    vector3_non_intersect_indices = np.nonzero(np.logical_not(vector3_intersect[:,0]))[0]

    intersect_points_1 =  vector1_coeff * vector1 + intersected_v0
    intersect_points_2 =  vector2_coeff * vector2 + intersected_v1
    intersect_points_3 =  vector3_coeff * vector3 + intersected_v2

    intersect_points_1[vector1_non_intersect_indices,:] = np.nan
    intersect_points_2[vector2_non_intersect_indices,:] = np.nan
    intersect_points_3[vector3_non_intersect_indices,:] = np.nan
    intersect_points_1 = intersect_points_1[:,0:2]
    intersect_points_2 =  intersect_points_2[:,0:2]
    intersect_points_3 =  intersect_points_3[:,0:2]

    intersect_points = np.concatenate((intersect_points_1,intersect_points_2,intersect_points_3),axis=1)
    intersect_points = intersect_points[np.isfinite(intersect_points)]
    intersect_points = np.reshape(intersect_points,(int(len(intersect_points)/4),4))


    for x in range(0,resolution_x):
        line_x = model_min_x + x * mm_per_pixel + 0.5 * mm_per_pixel

        intersects_line = np.logical_xor((intersect_points[:,0]>line_x),(intersect_points[:,2]>line_x))


        intersected_lines_indices = np.nonzero(intersects_line)[0]
        intersected_lines = intersect_points[intersected_lines_indices,:]

        layer_vectors = intersected_lines[:,2:4]-intersected_lines[:,0:2]
        layer_vector_origins = intersected_lines[:,0:2]
        intersect_points_1d = np.atleast_2d((line_x-layer_vector_origins[:,0])/layer_vectors[:,0]).T * layer_vectors + layer_vector_origins
        edge_pixels = np.round( (intersect_points_1d[:,1]-model_min_y) / mm_per_pixel  )
        edge_pixels = np.sort(edge_pixels)
        for i in range(0,int(len(edge_pixels)/2)):
            layer_array[ x,int(edge_pixels[i*2]) : int((edge_pixels[i*2+1]))] = 1#layer_z
    return layer_array