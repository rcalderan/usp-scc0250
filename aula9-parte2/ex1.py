import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
window = glfw.create_window(700, 700, "Esfera", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        uniform mat4 mat_transformation;
        void main(){
            gl_Position = mat_transformation * vec4(position,1.0);
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

import math
PI = 3.141592
r = 0.5 # raio
num_sectors = 30 # qtd de sectors (longitude)
num_stacks = 30 # qtd de stacks (latitude)

# grid sectos vs stacks (longitude vs latitude)
sector_step=(PI*2)/num_sectors # variar de 0 até 2π
stack_step=(PI)/num_stacks # variar de 0 até π

# Entrada: angulo de longitude, latitude, raio
# Saida: coordenadas na esfea
def F(u,v,r):
    x = r*math.sin(v)*math.cos(u)
    y = r*math.sin(v)*math.sin(u)
    z = r*math.cos(v)
    return (x,y,z)

# vamos gerar um conjunto de vertices representantes poligonos
# para a superficie da esfera.
# cada poligono eh representado por dois triangulos
vertices_list = []
for i in range(0,num_sectors): # para cada sector (longitude)
    
    for j in range(0,num_stacks): # para cada stack (latitude)
              
        u = i * sector_step # angulo setor
        v = j * stack_step # angulo stack
        
        un = 0 # angulo do proximo sector
        if i+1==num_sectors:
            un = PI*2
        else: un = (i+1)*sector_step
            
        vn = 0 # angulo do proximo stack
        if j+1==num_stacks:
            vn = PI
        else: vn = (j+1)*stack_step
        
        # verticies do poligono
        p0=F(u, v, r)
        p1=F(u, vn, r)
        p2=F(un, v, r)
        p3=F(un, vn, r)
        
        # triangulo 1 (primeira parte do poligono)
        vertices_list.append(p0)
        vertices_list.append(p2)
        vertices_list.append(p1)
        
        # triangulo 2 (segunda e ultima parte do poligono)
        vertices_list.append(p3)
        vertices_list.append(p1)
        vertices_list.append(p2)


total_vertices = len(vertices_list)
vertices = np.zeros(total_vertices, [("position", np.float32, 3)])
vertices['position'] = np.array(vertices_list)

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

glVertexAttribPointer(loc, 3, GL_FLOAT, False, stride, offset)

loc_color = glGetUniformLocation(program, "color")

glfw.show_window(window)

import math
import random
d = 0.0
glEnable(GL_DEPTH_TEST) ### importante para 3D

def multiplica_matriz(a,b):
    m_a = a.reshape(4,4)
    m_b = b.reshape(4,4)
    m_c = np.dot(m_a,m_b)
    c = m_c.reshape(1,16)
    return c

while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    ### apenas para visualizarmos a esfera rotacionando
    d -= 0.01 # modifica o angulo de rotacao em cada iteracao
    cos_d = math.cos(d)
    sin_d = math.sin(d)
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    mat_rotation_z = np.array([     cos_d, -sin_d, 0.0, 0.0, 
                                    sin_d,  cos_d, 0.0, 0.0, 
                                    0.0,      0.0, 1.0, 0.0, 
                                    0.0,      0.0, 0.0, 1.0], np.float32)
    
    mat_rotation_x = np.array([     1.0,   0.0,    0.0, 0.0, 
                                    0.0, cos_d, -sin_d, 0.0, 
                                    0.0, sin_d,  cos_d, 0.0, 
                                    0.0,   0.0,    0.0, 1.0], np.float32)
    
    mat_rotation_y = np.array([     cos_d,  0.0, sin_d, 0.0, 
                                    0.0,    1.0,   0.0, 0.0, 
                                    -sin_d, 0.0, cos_d, 0.0, 
                                    0.0,    0.0,   0.0, 1.0], np.float32)
    
    mat_transform = multiplica_matriz(mat_rotation_z,mat_rotation_x)
    mat_transform = multiplica_matriz(mat_rotation_y,mat_transform)

    loc = glGetUniformLocation(program, "mat_transformation")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)

    #glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    
    for triangle in range(0,len(vertices),3):
       
        random.seed( triangle )
        R = random.random()
        G = random.random()
        B = random.random()        
        glUniform4f(loc_color, R, G, B, 1.0)
        
        glDrawArrays(GL_TRIANGLES, triangle, 3)     

    
    glfw.swap_buffers(window)

glfw.terminate()