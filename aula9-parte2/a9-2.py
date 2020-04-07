"""
Richard Calderan
nusp: 3672382

30/03/2020

Exercicio prático: Desenhar cilindro e aplicar as transformações geométricas vinculadas a eventos de teclado

"""


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
h = 0.8 #altura
num_sectors = 30 # qtd de sectors 

# grid sectors
sector_step=(PI*2)/num_sectors # variar de 0 até 2π

# Entrada: angulo , altura, raio
# Saida: coordenadas na esfea
def F(a,h,r):
    x = r*math.cos(a)
    y = r*math.sin(a)
    z = h
    return (x,y,z)

# vamos gerar um conjunto de vertices representantes poligonos
# para a superficie do cilindro.
# cada poligono eh representado por dois triangulos
# é preciso também gerar a superficie de base do cilindo, que é dado por um triangulo para cada extremidade
vertices_list = []
for i in range(0,num_sectors): # para cada sector (longitude)
    
        u = i * sector_step # angulo setor
        
        un = 0 # angulo do proximo sector
        if i+1==num_sectors:
            un = PI*2
        else: un = (i+1)*sector_step
        

        pb = (0,0,-h/2)
        
        ph = (0,0,h/2)    
        # verticies do poligono
        p0=F(u, -h/2, r)
        p1=F(u, h/2, r)
        p2=F(un, -h/2, r)
        p3=F(un, h/2, r)

    #triangulo da base
        vertices_list.append(pb)
        vertices_list.append(p0)
        vertices_list.append(p2)
    #poligono
        # triangulo 1 (primeira parte do poligono)
        vertices_list.append(p0)
        vertices_list.append(p2)
        vertices_list.append(p1)
        
        # triangulo 2 (segunda e ultima parte do poligono)
        vertices_list.append(p3)
        vertices_list.append(p1)
        vertices_list.append(p2)

    #triangulo do topo
        vertices_list.append(ph)
        vertices_list.append(p1)
        vertices_list.append(p3)



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


#globais
x=0.0 #coordenadas para tranmslação
y=0.0
z=0.0

#escala
esc= 1.0

#angulos de rotação em seus eixos
ax=0
ay=0
az=0


#eventos 
def key_event(window,key,scancode,action,mods):
    global x,y,z, ax,ay,az,esc
    inc=0.05
    incEsc= 0.02
    incAng=0.08
    
    #transladar
    # tecla SETA CIMA
    if key == 265:
        y += inc        
    # tecla SETA BAIXO
    if key == 264:        
        y -= inc    
    # tecla SETA esquerda
    if key == 263:        
        x -= inc    
    # tecla SETA direita
    if key == 262:        
        x += inc

    #escala
    # tecla A
    if key == 65:        
        esc += incEsc 
    # tecla z 
    if key == 90:        
        esc -= incEsc 

    #rotacionar    
    # tecla SETA pageDown
    if key == 267:        
        ax -= incAng        
    # tecla SETA pageUp
    if key == 266:        
        ax += incAng
    # tecla SETA home
    if key == 268:        
        ay += incAng        
    # tecla SETA End
    if key == 269:        
        ay -= incAng        
    # tecla SETA insert
    if key == 260:        
        az += incAng        
    # tecla SETA delete
    if key == 261:        
        az -= incAng

glfw.set_key_callback(window,key_event)

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
    
    cos_x = math.cos(ax)
    sin_x = math.sin(ax)
    
    cos_y = math.cos(ay)
    sin_y = math.sin(ay)
    
    cos_z = math.cos(az)
    sin_z = math.sin(az)
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)

   #matriz de translação
    mat_transla = np.array([     1.0, 0.0, 0.0, x, 
                               0.0, 1.0, 0.0, y, 
                               0.0, 0.0, 1.0, z, 
                               0.0, 0.0, 0.0, 1.0], np.float32)

    #matriz escala                          
    mat_escala = np.array([     esc, 0.0, 0.0, 0.0, 
                                0.0, esc, 0.0, 0.0, 
                                0.0, 0.0, esc, 0.0, 
                                0.0, 0.0, 0.0, 1.0], np.float32)

    #matrizes de rotação
    mat_rotation_z = np.array([     cos_z, -sin_z, 0.0, 0.0, 
                                    sin_z,  cos_z, 0.0, 0.0, 
                                    0.0,      0.0, 1.0, 0.0, 
                                    0.0,      0.0, 0.0, 1.0], np.float32)
    
    mat_rotation_x = np.array([     1.0,   0.0,    0.0, 0.0, 
                                    0.0, cos_x, -sin_x, 0.0, 
                                    0.0, sin_x,  cos_x, 0.0, 
                                    0.0,   0.0,    0.0, 1.0], np.float32)
    
    mat_rotation_y = np.array([     cos_y,  0.0, sin_y, 0.0, 
                                    0.0,    1.0,   0.0, 0.0, 
                                    -sin_y, 0.0, cos_y, 0.0, 
                                    0.0,    0.0,   0.0, 1.0], np.float32)

    mat_transform = multiplica_matriz(mat_rotation_z,mat_rotation_x)
    mat_transform = multiplica_matriz(mat_rotation_y,mat_transform)    
    mat_transform = multiplica_matriz(mat_transla,mat_transform)
    
    mat_transform = multiplica_matriz(mat_transform,mat_escala)

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