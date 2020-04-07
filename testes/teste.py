import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

import gpxpy 
import gpxpy.gpx 



gpx_file = open('gpx.gpx', 'r') 

gpx = gpxpy.parse(gpx_file)
points =gpx.tracks[0].segments[0].points
print(len(points))

def selectPoints(pts):
    selected = []
    lP=pts[0]
    dif=0
    sum=0
    for p in pts:
        dif = p.latitude + lP.latitude*(-1)
        if dif<0:
            dif=-1*dif
        dif*=100000
        sum+=dif
        if sum>20:
            #print ('Point at ({0},{1}) -> {2}'.format(p.latitude, p.longitude, sum) )
            selected.append(p)
            sum=0
        #if dif>5:
        lP=p
    return selected
#selectPoints(points)
points = selectPoints(points)
print(len(points))


glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(800, 800, "teste", None, None)
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
program = glCreateProgram()
vertex = glCreateShader(GL_VERTEX_SHADER)
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

vertices = np.zeros(len(points), [("position", np.float32, 2)])
i=0
aj=-1
mx=0
my=0
la=0
lo=0
for p in points:

    la=p.latitude+22.017
    lo=p.longitude+47.896
    mx+=la
    my+=lo
    vertices['position'][i] = ( (la,lo))
    
    #print ('Point at ({0},{1}) -> {2}'.format(p.latitude, p.longitude, p.elevation) )

    i+=1
    #print((p.latitude+22))
    #print(p.longitude*0.01)

#vertices['position'][0]=((0,0))
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



t_x=0 #mx/len(points)
t_y=-4#my/len(points)
scala =100
def key_event(window, key, scancode, action, mods):
    global t_x,t_y,scala
    # inicia a deformação da amola
    
    if key == 264:
        t_y-= 0.001

    if key == 265:
        t_y+= 0.001
    if key == 262:
        t_x+= 0.001
    if key == 263:
        t_x-= 0.001
        
    if key == 267:
        scala-= 0.5
    if key == 266:
        scala+= 0.5

glfw.set_key_callback(window, key_event)



glfw.show_window(window)


def multiplica_matriz(a, b):
    m_a = a.reshape(4, 4)
    m_b = b.reshape(4, 4)
    m_c = np.dot(m_a, m_b)
    c = m_c.reshape(1, 16)
    return c


angulo = 180.0

cos = math.cos(math.radians(angulo))
sin = math.sin(math.radians(angulo))
while not glfw.window_should_close(window):

    # a mola deve retornar ao estado original de deformação quando a seta não está sendo pressionada
    
    mat_transform = np.array([1.0, 0.0, 0.0, 0,
                          0.0, 1.0, 0.0, 0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0], np.float32)
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    mat_rotation = np.array([cos, -sin, 0.0, 0.0,
                             sin, cos, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0], np.float32)


    mat_translation = np.array([1.0, 0.0, 0.0, t_x,
                                0.0, 1.0, 0.0, t_y,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0], np.float32)

    mat_scala = np.array([scala, 0.0, 0.0, 0.0,
                          0.0, scala, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0], np.float32)

    mat_transform = multiplica_matriz(mat_transform, mat_translation)

    mat_transform = multiplica_matriz(mat_transform, mat_rotation)

    mat_transform = multiplica_matriz(mat_transform, mat_scala)


    loc = glGetUniformLocation(program, "mat")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)

    glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
    

    glfw.swap_buffers(window)

glfw.terminate()
