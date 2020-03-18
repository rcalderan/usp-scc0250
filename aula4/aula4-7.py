import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
window = glfw.create_window(720, 600, "Cores", None, None)
glfw.make_context_current(window)

def key_event(window,key,scancode,action,mods):
    print('[key event] key=',key)
    print('[key event] scancode=',scancode)
    print('[key event] action=',action)
    print('[key event] mods=',mods)
    print('-------')
    
glfw.set_key_callback(window,key_event)

def mouse_event(window,button,action,mods):
    print('[mouse event] button=',button)
    print('[mouse event] action=',action)
    print('[mouse event] mods=',mods)
    print('-------')
glfw.set_mouse_button_callback(window,mouse_event)

vertex_code = """
        attribute vec2 position;
        void main(){
            gl_Position = vec4(position,0.0,1.0);
        }
        """
fragment_code_old = """
        void main(){
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
        """
fragment_code = """
        uniform vec4 color;
        void main(){
            gl_FragColor = color;
        }
        """
# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")

glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)

# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)

#a magica começa aqui# preparando espaço para 3 vértices usando 2 coordenadas (x,y)
vertices = np.zeros(3, [("position", np.float32, 2)])

vertices['position'] = [
                            ( 0.0, 0.0), # vertice 0
                            (+0.5,+0.5), # vertice 1
                            (+0.5, 0.0) # vertice 2
]
# Request a buffer slot from GPU
buffer = glGenBuffers(1)
# Make this buffer the default one
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Upload data
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, buffer)

# Bind the position attribute
# --------------------------------------
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)

loc = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc)

glVertexAttribPointer(loc, 2, GL_FLOAT, False, stride, offset)

loc_color = glGetUniformLocation(program, "color")
R = 1.0
G = 0.0
B = 0.0

glfw.show_window(window)

r=0
rr=0.01
g=.8
gg=0.02
b=0.5
bb=0.03
while not glfw.window_should_close(window):

    glfw.poll_events() 
    r+=rr
    if r>=1:
        rr=-0.01
    elif r<=0:
        rr=0.01

    g+=gg
    if g>=1:
        gg=-0.02
    elif g<=0:
        gg=0.02
        
    b+=bb
    if b>=1:
        bb=-0.03
    elif b<=0:
        bb=0.03
    
    glClear(GL_COLOR_BUFFER_BIT) 
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))
    glUniform4f(loc_color, r, g, b, 1.0) ### modificando a cor do objeto!

    glfw.swap_buffers(window)

glfw.terminate()
