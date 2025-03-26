def define_shader():
    vertex_source = """#version 330 core
        in vec3 position;
        in vec3 tex_coords;
        out vec3 texture_coords;

        uniform WindowBlock 
        {                       // This UBO is defined on Window creation, and available
            mat4 projection;    // in all Shaders. You can modify these matrixes with the
            mat4 view;          // Window.view and Window.projection properties.
        } window;  
        uniform mat4 offset;

        void main()
        {
            gl_Position = window.projection *  window.view *offset*vec4(position, 1);
            texture_coords = tex_coords;
        }
    """

    fragment_source = """#version 330 core
        in vec3 texture_coords;
        out vec4 final_colors;

        uniform sampler2D our_texture;

        void main()
        {
            final_colors = texture(our_texture, texture_coords.xy);
        }
    """
    return vertex_source, fragment_source