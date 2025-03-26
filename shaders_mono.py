def define_shader():
    vertex_source = """#version 330 core
        in vec3 position;
        in vec3 normals;
        out vec3 colors;

        uniform WindowBlock 
        {                       // This UBO is defined on Window creation, and available
            mat4 projection;    // in all Shaders. You can modify these matrixes with the
            mat4 view;          // Window.view and Window.projection properties.
        } window;  
        uniform mat4 offset;
        uniform vec3 color_in;

        void main()
        {
            gl_Position = window.projection * window.view *offset*vec4(position, 1);
            float val = dot(normals,vec3(0,0.9,0.2));
            colors = val * color_in;
        }
    """

    fragment_source = """#version 330 core
        in vec3 colors;
        out vec4 final_colors;

        void main()
        {
            final_colors = vec4(colors,1);
        }
    """
    return vertex_source, fragment_source