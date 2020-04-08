import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(800, 800, "Trabalho1 - Mola 2D", None, None)
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
numberOfVertices = 128
vertices = np.zeros(numberOfVertices, [("position", np.float32, 2)])


'''
1. Desenhar uma Mola 2D no centro da cena. (Peso: 2.0)
a. Os vértices devem ser gerados automaticamente por uma
função.
aqui
'''


def criaMola1():  # função para criação de mola. Basicamente um circulo com varias revoluções que vamos esticando
    an = 90
    tam = 0
    for i in range(numberOfVertices):
        px = math.sin(math.radians(an))
        vertices['position'][i] = (px, tam)
        an += 30
        tam += 1/numberOfVertices

def criaMola(tam):  # função para criação de mola. Basicamente um circulo com varias revoluções que vamos esticando
    an = 90
    py=-tam/2
    for i in range(numberOfVertices):
        px = math.sin(math.radians(an))
        vertices['position'][i] = (px/8, py)
        an += 30
        py += tam/numberOfVertices

tamanho=0.5
criaMola(tamanho)

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


'''
2. Aplicar transformações geométricas na mola:
a. Ao segurar na seta para baixo do teclado a mola deve
comprimir em relação ao tempo pressionado (Escala). (Peso:
2.0)

aplicamos a transformação de ESCALA para simular a compressão da mola
'''


# globais
xi=0
yi=tamanho/2
t_x = xi
t_y = yi
angulo = 0.0

cos = math.cos(math.radians(angulo))
sin = math.sin(math.radians(angulo))

deform = 1

keypressed = 0
isMoving = False

sentido = np.random.randint(2)


def key_event(window, key, scancode, action, mods):
    global deform, keypressed, isMoving

    # inicia a deformação da amola
    if key == 264:
        if not isMoving:
            keypressed = action 
            if deform > 0.15:
                deform -= 0.02
        else:
            keypressed = 0
            isMoving = True


glfw.set_key_callback(window, key_event)

x0 = 0

def calcular_coordenadas_x_y():
    global t_x, t_y, xi,yi,cos, sin, sentido, angulo, isMoving, deform
    # intensidade da deformação pode ser:
    variant = 0.4
    intensit = .3
    # print(angulo)
    if isMoving:  # move somente quando o gatilho for disparado
        print(sentido)
        if sentido > 0:
            angulo += variant
        else:
            angulo -= variant
        if angulo < 0:
            isMoving = False
            sentido = np.random.randint(2)
            angulo = 1
            return
        if angulo >= 180:
            isMoving = False
            sentido = np.random.randint(2)
            angulo = 179
            return
        cos = math.cos(math.radians(angulo))
        sin = math.sin(math.radians(angulo))
        t_x = xi+cos*intensit
        t_y = yi+sin*intensit
    else:
        xi=0
        yi=0


glfw.show_window(window)


def multiplica_matriz(a, b):
    m_a = a.reshape(4, 4)
    m_b = b.reshape(4, 4)
    m_c = np.dot(m_a, m_b)
    c = m_c.reshape(1, 16)
    return c


while not glfw.window_should_close(window):

    # a mola deve retornar ao estado original de deformação quando a seta não está sendo pressionada
    
    if deform < 1 and keypressed == 0:
        deform += 0.03
        if deform >= 0.25:
            sentido = np.random.randint(2)
            isMoving = True

    calcular_coordenadas_x_y()
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

    mat_scala = np.array([1.0, 0.0, 0.0, 0.0,
                          0.0, deform, 0.0, 0.0,
                          0.0, 0.0, 1.0, 0.0,
                          0.0, 0.0, 0.0, 1.0], np.float32)

    mat_transform = multiplica_matriz(mat_scala, mat_translation)

    mat_transform = multiplica_matriz(mat_transform, mat_rotation)

    #mat_transform = multiplica_matriz(mat_transform, )


    loc = glGetUniformLocation(program, "mat")
    glUniformMatrix4fv(loc, 1, GL_TRUE, mat_transform)

    glDrawArrays(GL_LINE_STRIP, 0, len(vertices))

    glfw.swap_buffers(window)

glfw.terminate()
