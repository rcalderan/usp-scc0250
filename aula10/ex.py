import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
altura = 700
largura = 700
window = glfw.create_window(largura, altura, "Cameras - Cubo", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
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

# preparando espaço para n vértices usando 2 coordenadas (x,y)
vertices = np.zeros(5464, [("position", np.float32, 3)])

# preenchendo as coordenadas de cada vértice
primitives_list = [
    
    ### CUBO 1
    # Face 1 do Cubo 1 (vértices do quadrado)
    (-0.2, -0.2, +0.2),
    (+0.2, -0.2, +0.2),
    (-0.2, +0.2, +0.2),
    (+0.2, +0.2, +0.2),

    # Face 2 do Cubo 1
    (+0.2, -0.2, +0.2),
    (+0.2, -0.2, -0.2),         
    (+0.2, +0.2, +0.2),
    (+0.2, +0.2, -0.2),
    
    # Face 3 do Cubo 1
    (+0.2, -0.2, -0.2),
    (-0.2, -0.2, -0.2),            
    (+0.2, +0.2, -0.2),
    (-0.2, +0.2, -0.2),

    # Face 4 do Cubo 1
    (-0.2, -0.2, -0.2),
    (-0.2, -0.2, +0.2),         
    (-0.2, +0.2, -0.2),
    (-0.2, +0.2, +0.2),

    # Face 5 do Cubo 1
    (-0.2, -0.2, -0.2),
    (+0.2, -0.2, -0.2),         
    (-0.2, -0.2, +0.2),
    (+0.2, -0.2, +0.2),
    
    # Face 6 do Cubo 1
    (-0.2, +0.2, +0.2),
    (+0.2, +0.2, +0.2),           
    (-0.2, +0.2, -0.2),
    (+0.2, +0.2, -0.2),


    #### CUBO 2
    # Face 1 do Cubo 2 (vértices do quadrado)
    (+0.1, +0.1, -0.5),
    (+0.5, +0.1, -0.5),
    (+0.1, +0.5, -0.5),
    (+0.5, +0.5, -0.5),

    # Face 2 do Cubo 2
    (+0.5, +0.1, -0.5),
    (+0.5, +0.1, -0.9),         
    (+0.5, +0.5, -0.5),
    (+0.5, +0.5, -0.9),
    
    # Face 3 do Cubo 2
    (+0.5, +0.1, -0.9),
    (+0.1, +0.1, -0.9),            
    (+0.5, +0.5, -0.9),
    (+0.1, +0.5, -0.9),

    # Face 4 do Cubo 2
    (+0.1, +0.1, -0.9),
    (+0.1, +0.1, -0.5),         
    (+0.1, +0.5, -0.9),
    (+0.1, +0.5, -0.5),

    # Face 5 do Cubo 2
    (+0.1, +0.1, -0.9),
    (+0.5, +0.1, -0.9),         
    (+0.1, +0.1, -0.5),
    (+0.5, +0.1, -0.5),
    
    # Face 6 do Cubo 2
    (+0.1, +0.5, -0.5),
    (+0.5, +0.5, -0.5),           
    (+0.1, +0.5, -0.9),
    (+0.5, +0.5, -0.9),

     # Face 1 a Piramide (vértices do quadrado)
    (+0.5, -0.6, +0.5),
    (-0.5, 0.6, +0.5),
    (0.5, +0.6, -0.5),
    (-0.5, +0.6, -0.5),

    # Face 2 da Piramide (triangulo)
    (-0.5, 0.6, +0.5),
    (+0.5, -0.6, +0.5),         
    (+0.0, +1.6, +0.0),
    
    # Face 3 da Piramide
    (+0.5, -0.6, +0.5),
    (0.5, +0.6, -0.5),       
    (+0.0, +1.6, +0.0),

    # Face 4 da Piramide
    (0.5, +0.6, -0.5),
    (-0.5, +0.6, -0.5),      
    (+0.0, +1.6, +0.0),

    # Face 5 da Piramide
    (-0.5, +0.6, -0.5),
    (-0.5, +0.6, 0.5),      
    (+0.0, +1.6, +0.0)]

vertices_list = []
for i in range(len(primitives_list)):
    vertices_list.append(primitives_list[i])

#Preparando os vertices da esfera
PI = 3.141592
r = 0.2 # raio
num_sectors = 30 # qtd de sectors (longitude)
num_stacks = 30 # qtd de stacks (latitude)

# grid sectos vs stacks (longitude vs latitude)
sector_step=(PI*2)/num_sectors # variar de 0 até 2π
stack_step=(PI)/num_stacks # variar de 0 até π

# Entrada: angulo de longitude, latitude, raio
# Saida: coordenadas na esfea
def F(u,v,r):
    offset_x =0.6
    offset_y =0
    offset_z =0

    x = r*math.sin(v)*math.cos(u)+offset_x
    y = r*math.sin(v)*math.sin(u)+offset_y
    z = r*math.cos(v)+offset_z
    return (x,y,z)

# vamos gerar um conjunto de vertices representantes poligonos
# para a superficie da esfera.
# cada poligono eh representado por dois triangulos

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

##--------


# Adicionar as primitivas
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


cameraPos   = glm.vec3(0.0,  0.0,  1.0)
cameraFront = glm.vec3(0.0,  0.0, -1.0)
cameraUp    = glm.vec3(0.0,  1.0,  0.0)


def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp
    
    cameraSpeed = 0.01
    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)

def model():
    mat_model = glm.mat4(1.0) # matriz identidade
    mat_model = np.array(mat_model)    
    return mat_model

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp)
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 100.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) ### importante para 3D


def desenha_cubo1():
    # DESENHANDO O CUBO 1 (vértices de 0 até 23)
    for i in range(0,24,4): # incremento de 4 em 4
        R = (i+1)/24
        G = (i+2)/24
        B = (i+3)/24
        glUniform4f(loc_color, R, G, B, 1.0) ### definindo uma cor qualquer com base no i
        glDrawArrays(GL_TRIANGLE_STRIP, i, 4)
    
def desenha_cubo2():
    # DESENHANDO O CUBO 2 (vértices de 24 até 47)
    for i in range(24,48,4): # incremento de 4 em 4
        R = (i+1)/48
        G = (i+2)/48
        B = (i+3)/48
        glUniform4f(loc_color, R, G, B, 1.0) ### definindo uma cor qualquer com base no i
        glDrawArrays(GL_TRIANGLE_STRIP, i, 4)

def desenha_priramide():
    # DESENHANDO A PIRAMIDE (vértices de 47 até 55)

    #base
    glUniform4f(loc_color, 1, 0, 0, 1.0) ### vermelho
    glDrawArrays(GL_TRIANGLE_STRIP, 48, 4)
    
    for i in range(52,64,3): # incremento de 3 em 3
        R = (i+1)/64
        G = (i+2)/64
        B = (i+3)/64
        glUniform4f(loc_color, R, G, B, 1.0) ### definindo uma cor qualquer com base no i
        glDrawArrays(GL_TRIANGLE_STRIP, i, 3)


import random
def desenha_esfera():
    # DESENHANDO A ESFERA (vértices de 48 até 5449)
    
    for triangle in range(64,5464,3):
       
        random.seed( triangle )
        R = random.random()
        G = random.random()
        B = random.random()        
        glUniform4f(loc_color, R, G, B, 1.0)
        
        glDrawArrays(GL_TRIANGLES, triangle, 3) 




while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    

    # computando e enviando matrizes Model, View e Projection para a GPU
    mat_model = model()
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_FALSE, mat_model)
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    

    # desenhando objetos
   
    desenha_cubo1()
    
    desenha_cubo2()

    desenha_priramide()

    desenha_esfera()

    
    glfw.swap_buffers(window)

glfw.terminate()
