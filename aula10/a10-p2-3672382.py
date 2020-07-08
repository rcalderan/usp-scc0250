"""
Richard Calderan  - 3672382
Exercicio alua 10 parte 2

"""
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
altura = 800
largura = 800
window = glfw.create_window(largura, altura, "Cameras - Matriz Model", None, None)
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

num_cubos = 5 # cinco cubos
num_piramides = 2 # cinco cubos
vertices = np.zeros(num_cubos*24+num_piramides*16, [("position", np.float32, 3)])

def get_cubo():
    cubo = [
    # Face 1
    (-0.1, -0.1, +0.1),
    (+0.1, -0.1, +0.1),
    (-0.1, +0.1, +0.1),
    (+0.1, +0.1, +0.1),

    # Face 2
    (+0.1, -0.1, +0.1),
    (+0.1, -0.1, -0.1),         
    (+0.1, +0.1, +0.1),
    (+0.1, +0.1, -0.1),
    
    # Face 3
    (+0.1, -0.1, -0.1),
    (-0.1, -0.1, -0.1),            
    (+0.1, +0.1, -0.1),
    (-0.1, +0.1, -0.1),

    # Face 4
    (-0.1, -0.1, -0.1),
    (-0.1, -0.1, +0.1),         
    (-0.1, +0.1, -0.1),
    (-0.1, +0.1, +0.1),

    # Face 5
    (-0.1, -0.1, -0.1),
    (+0.1, -0.1, -0.1),         
    (-0.1, -0.1, +0.1),
    (+0.1, -0.1, +0.1),
    
    # Face 6
    (-0.1, +0.1, +0.1),
    (+0.1, +0.1, +0.1),           
    (-0.1, +0.1, -0.1),
    (+0.1, +0.1, -0.1)]
    
    return cubo


def get_piramide():
    piramide = [
    # Face 1
    (-0.1, -0.1, -0.1), 
    (-0.1, -0.1, +0.1),
    (+0.1, -0.1, -0.1),
    (+0.1, -0.1, +0.1),

    # Face 2
    (-0.1, -0.1, +0.1),
    (+0.1, -0.1, +0.1),        
    (+0.05, +0.05, +0.05),
    
    # Face 3
    (+0.1, -0.1, -0.1),
    (-0.1, -0.1, -0.1),        
    (+0.05, +0.05, +0.05),

    # Face 4
    (-0.1, -0.1, -0.1),
    (-0.1, -0.1, +0.1),         
    (+0.05, +0.05, +0.05),

    # Face 5
    (-0.1, -0.1, -0.1),
    (+0.1, -0.1, -0.1),         
    (+0.05, +0.05, +0.05)]
    
    return piramide


# preenchendo o vetor de vertices com todos os cubos (num_cubos)
cubos = get_cubo() # cubo numero 1
for i in range(1,num_cubos): # pegando o restante dos outros cubos
    # pegando um novo cubo
    vert_cubo = get_cubo()
    
    # adicionando os vertices do cubo no nosso vertor de vertices
    cubos = np.concatenate((cubos, vert_cubo), axis=0)
objetos=[]
for i in range(len(cubos)):
    objetos.append(cubos[i])

piramides = get_piramide()
for i in range(1,num_piramides): # pegando o restante dos outros cubos
    # pegando um novo cubo
    vert_p = get_piramide()
    
    # adicionando os vertices do cubo no nosso vertor de vertices
    piramides = np.concatenate((piramides, vert_p), axis=0)
for i in range(len(piramides)):
    objetos.append(piramides[i])
    
vertices['position'] = objetos

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


def model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    angle = math.radians(angle)
    
    matrix_transform = glm.mat4(1.0) # instanciando uma matriz identidade
       
    # aplicando rotacao
    matrix_transform = glm.rotate(matrix_transform, angle, glm.vec3(r_x, r_y, r_z))
        
  
    # aplicando translacao
    matrix_transform = glm.translate(matrix_transform, glm.vec3(t_x, t_y, t_z))    
    
    # aplicando escala
    matrix_transform = glm.scale(matrix_transform, glm.vec3(s_x, s_y, s_z))
    
    matrix_transform = np.array(matrix_transform).T # pegando a transposta da matriz (glm trabalha com ela invertida)
    
    return matrix_transform

def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
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

def desenha_cubo(index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    cores_face = [
        [1.0, 0.0, 0.0], # R, G, B
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],        
        [1.0, 0.0, 1.0],
    ]
    
    # DESENHANDO O CUBO
    face = 0
    for i in range(index*24,(index+1)*24,4): # incremento de 4 em 4 (desenhando cada face)
        R = cores_face[face][0]
        G = cores_face[face][1]
        B = cores_face[face][2]
        glUniform4f(loc_color, R, G, B, 1.0) ### definindo uma cor
        glDrawArrays(GL_TRIANGLE_STRIP, i, 4) ## renderizando
        face+=1

def desenha_piramide(index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z):
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    cores_face = [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0], # R, G, B
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],  
        [0.0, 1.0, 0.0],  
        [0.0, 1.0, 0.0],  
        [0.0, 1.0, 0.0],  
        [0.0, 1.0, 0.0],  
        [0.0, 1.0, 0.0],  
    ]
    
    # DESENHANDO 
    # #base
    R = cores_face[0][0]
    G = cores_face[0][1]
    B = cores_face[0][2]
    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLE_STRIP, index*16, 4) ## renderizando
    
    R = cores_face[1][0]
    G = cores_face[1][1]
    B = cores_face[1][2]
    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLE_STRIP, index*16+4, 3) ## renderizando

    print(index*16+4)
    R = cores_face[2][0]
    G = cores_face[2][1]
    B = cores_face[2][2]
    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLE_STRIP, index*16+8, 3) ## renderizando
    
    print(index*16+8)
    R = cores_face[3][0]
    G = cores_face[3][1]
    B = cores_face[3][2]
    glUniform4f(loc_color, R, G, B, 1.0)
    glDrawArrays(GL_TRIANGLE_STRIP, index*16+11, 3) ## renderizando
    
    print(index*16+11)
    
    # #faces triangulares
    # face = 1
    # for i in range(index*16+1,(index+1)*16+face*3,3): # incremento de 3 em 3 
    #     print(face)
    #     R = cores_face[face][0]
    #     G = cores_face[face][1]
    #     B = cores_face[face][2]
    #     glUniform4f(loc_color, R, G, B, 1.0) ### definindo uma cor
    #     glDrawArrays(GL_TRIANGLE_STRIP, i, 4) ## renderizando
    #     face+=1     

while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    #glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    # computando e enviando matrizes Model, View e Projection para a GPU
    
    # temos uma matriz model por objeto!
    object_index=0   
    # angulo de rotacao e eixos
    angle=0.0; r_x=0.0; r_y=0.0; r_z=1.0
    # translacao
    t_x=0.5; t_y=0.0; t_z=0.0
    # escala
    s_x=1.0; s_y=1.0; s_z=1.0
    desenha_cubo(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    
    
    object_index=1
    # angulo de rotacao e eixos
    angle=0.0; r_x=0.0; r_y=0.0; r_z=1.0
    # translacao
    t_x=-0.5; t_y=0.0; t_z=0.0
    # escala
    s_x=1.0; s_y=1.0; s_z=1.0
    desenha_cubo(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    
    
    object_index=2
    # angulo de rotacao e eixos
    angle=-20.0; r_x=0.0; r_y=0.0; r_z=1.0
    # translacao
    t_x=-0.2; t_y=0.0; t_z=-0.7
    # escala
    s_x=0.2; s_y=5.0; s_z=5.0
    #s_x=1.0; s_y=1.0; s_z=1.0
    desenha_cubo(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    
    
    object_index=3
    # angulo de rotacao e eixos
    angle=20.0; r_x=0.0; r_y=0.0; r_z=1.0
    # translacao
    t_x=+0.2; t_y=0.0; t_z=-0.7
    # escala
    s_x=0.2; s_y=5.0; s_z=5.0
    #s_x=1.0; s_y=1.0; s_z=1.0
    desenha_cubo(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    
    
    object_index=4
    # angulo de rotacao e eixos
    angle=180.0; r_x=1.0; r_y=0.0; r_z=0.0
    # translacao
    t_x=0.0; t_y=0.4; t_z=0.0
    # escala
    s_x=15.0; s_y=0.1; s_z=15.0
    desenha_cubo(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)

    #agora desenhar as piramides
    object_index=5
    # angulo de rotacao e eixos
    angle=0.0; r_x=1.0; r_y=0.0; r_z=0.0
    # translacao
    t_x=0.0; t_y=0.0; t_z=0.0
    # escala
    s_x=1.0; s_y=1.0; s_z=1.0
    desenha_piramide(object_index, angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)    
    
    glfw.swap_buffers(window)

glfw.terminate()