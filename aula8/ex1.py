import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
window = glfw.create_window(800, 800, "Transformação Geométrica", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec2 position;
        uniform mat4 mat;
        void main(){
            gl_Position = mat * vec4(position,0.0,1.0);
        }
        """
fragment_code = """
        void main(){
            gl_FragColor = vec4(1.0,0.0,0.0,1.0);
        }
        """

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)

# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

# Compile shaders
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

# preparando espaço para 3 vértices usando 2 coordenadas (x,y)
vertices = np.zeros(3, [("position", np.float32, 2)])

# preenchendo as coordenadas de cada vértice
vertices['position'] = [
                            (+0.15, +0.00), 
                            (-0.05, -0.05), 
                            (-0.05, +0.05)
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

# translacao e rotacao
t_x = 0.0
t_y = 0.0
angulo = 0.0

x_inc = 0.0
y_inc = 0.0
r_inc = 0.0


def key_event(window,key,scancode,action,mods):
    global x_inc, y_inc, r_inc
    
    # tecla SETA CIMA
    if key == 265:
        y_inc += 0.0001
        x_inc += 0.0001
        
    
    # tecla SETA BAIXO
    if key == 264:
        y_inc -= 0.0001
        x_inc -= 0.0001
        
    
    # tecla SETA DIREITA    
    if key == 262: 
        r_inc -= 0.1
    
    # tecla SETA ESQUERDA
    if key == 263:
        r_inc += 0.1
    

    
glfw.set_key_callback(window,key_event)

def verifica_fronteiras():
    global t_x, t_y
    if (t_x < -1.2 or t_x > 1.2 or t_y < -1.2 or t_y > 1.2):
        t_x = -t_x
        t_y = -t_y

def calcular_coordenadas_x_y():
    global t_x, t_y
    
    t_x += x_inc * math.cos( math.radians(angulo) )
    t_y += y_inc * math.sin( math.radians(angulo) )

glfw.show_window(window)

def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c


while not glfw.window_should_close(window):

    verifica_fronteiras()
    calcular_coordenadas_x_y()
    
    angulo += r_inc
    
    c = math.cos( math.radians(angulo) )
    s = math.sin( math.radians(angulo) )
    
    glfw.poll_events() 

    
    glClear(GL_COLOR_BUFFER_BIT) 
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    
    #Draw Triangle
    
    mat_rotation = np.array([  c  , -s , 0.0, 0.0, 
                               s  , c  , 0.0, 0.0, 
                               0.0, 0.0, 1.0, 0.0, 
                               0.0, 0.0, 0.0, 1.0], np.float32)
    
    mat_translation = np.array([  1.0, 0.0, 0.0, t_x, 
                                  0.0, 1.0, 0.0, t_y, 
                                  0.0, 0.0, 1.0, 0.0, 
                                  0.0, 0.0, 0.0, 1.0], np.float32)
    


    mat_transform = multiplica_matriz(mat_translation,mat_rotation)
    

    
    loc = glGetUniformLocation(program, "mat")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)
    
    glDrawArrays(GL_TRIANGLES, 0, len(vertices))

    glfw.swap_buffers(window)

glfw.terminate()