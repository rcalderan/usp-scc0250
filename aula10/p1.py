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

vertices = np.zeros(48, [("position", np.float32, 3)])

# preenchendo as coordenadas de cada vértice
vertices['position'] = [
    
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
    (+0.5, +0.5, -0.9)]

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

    
    glfw.swap_buffers(window)

glfw.terminate()
