"""
USP  Computação Gráfica - SCC0250 2020/1
Prof. Ricardo Marcondes Marcacini

Richard Calderan - 3672382

Trabalho 2 

obs:
Requisito 5- ao invés do terreno ter 2 texturas, eu optei por deixar os outros objetos com mais texturas
O "gollum", por exemplo tem 4

requisito 9: Eu gostaria de limitar o movimento da camera criando uma função que pegasse a sua posição e comparasse com um
o vertice mais proximo dela comparando suas coordenadas no eixo Y, mas infelizmente esta abordagem se demostrou ineficiente. Eu poderia 
ter escolhido um plano ou uma caixa para o terreno externo, seria mais facil...

tecla O e L avança a câmera mais rápido.

"""


import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image


glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
altura = 1600
largura = 1200
window = glfw.create_window(largura, altura, "Malhas e Texturas", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        varying vec2 out_texture;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
        }
        """

fragment_code = """
        uniform vec4 color;
        varying vec2 out_texture;
        uniform sampler2D samplerTexture;
        
        void main(){
            vec4 texture = texture2D(samplerTexture, out_texture);
            gl_FragColor = texture;
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

def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue
        


        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])


        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])
        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)
            faces.append((face, face_texture, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces

    return model

glEnable(GL_TEXTURE_2D)
qtd_texturas = 20
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

vertices_list = []
textures_coord_list = []

modelo = load_model_from_file('casa.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo casa.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo casa.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(0,'Texture/casa.jpg')

modelo = load_model_from_file('assets/cama/cama.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo cama.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo cama.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(1,'assets/cama/cama.png')

modelo = load_model_from_file('assets/sky/sky.obj')

#armazena os vertices do chao para vinculá-los a um sistema de colisão
maxVetexes=modelo['vertices']

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo sky.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo sky.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(2,'assets/sky/sky.jpg')

faces_visitadas = []
modelo = load_model_from_file('assets/chao/chao.obj')
#armazena os vertices do chao para vinculá-los a um sistema de colisão
chaoVert=[]  

chaoSize = 451
off_x=100.0
off_y=-15.0
off_z=45.0

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo chao.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    if face[2] not in faces_visitadas:
        print(face[2], 'face vertice inicial: ',len(vertices_list))
        faces_visitadas.append(face[2])
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
        chaoVert.append(modelo['vertices'][vertice_id-1])
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo chao.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(3,'assets/chao/terra.jpg')
load_texture_from_file(4,'assets/chao/field.jpg')

modelo = load_model_from_file('assets/Tree 02/Tree.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo Tree.obj. Vertice inicial:',len(vertices_list))
faces_visitadas = []
for face in modelo['faces']:
    if face[2] not in faces_visitadas:
        print(face[2], 'face vertice inicial: ',len(vertices_list))
        faces_visitadas.append(face[2])
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo chao.Tree. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(5,'assets/Tree 02/bark_0004.jpg')
load_texture_from_file(6,'assets/Tree 02/DB2X2_L01.png')

modelo = load_model_from_file('assets/elsa/elsa.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo elsa.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo elsa.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(7,'assets/elsa/elsa.png')

faces_visitadas = []
modelo = load_model_from_file('assets/dog/dog.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo dog.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    if face[2] not in faces_visitadas:
        print(face[2], 'face vertice inicial: ',len(vertices_list))
        faces_visitadas.append(face[2])
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo dog.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
#load_texture_from_file(8,'assets/dog/Tex_0093_0.tga')
#load_texture_from_file(9,'assets/dog/Tex_0369_0.tga')
load_texture_from_file(8,'assets/dog/Tex_0552_7.tga')


faces_visitadas = []
modelo = load_model_from_file('assets/gollum/gollum.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo gollum.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    if face[2] not in faces_visitadas:
        print(face[2], 'face vertice inicial: ',len(vertices_list))        
        faces_visitadas.append(face[2])
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo gollum.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(9,'assets/gollum/4e4e0374.PNG')
load_texture_from_file(10,'assets/gollum/553e267b.PNG')
load_texture_from_file(11,'assets/gollum/2093f02c.PNG')
load_texture_from_file(12,'assets/gollum/c581eba0.PNG')

modelo = load_model_from_file('assets/ring/ring.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo ring.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo ring.obj. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(13,'assets/ring/ring.png')

# Request a buffer slot from GPU
buffer = glGenBuffers(2)

vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list

# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)


def desenha_casa():
    # aplica a matriz model
    # rotacao
    angle = 90.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = 25.0; t_y = -12.0; t_z = -150.0
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 0, 1770) ## renderizando

def desenha_cama():
    # aplica a matriz model
    # rotacao
    angle = 180.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = 140.0; t_y = -11.0; t_z = 3.5
    
    # escala
    s_x = 5.0; s_y = 5.0; s_z = 5.0
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 1770, 3156-1770) ## renderizando


def desenha_sky():
    # aplica a matriz model
    # rotacao
    angle = 0.0
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = 0.0; t_y = -50.0; t_z = 0.0
    
    # escala
    s_x = 1500.0; s_y = 1500.0; s_z = 1500.0
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 2)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 3156, 4596-3156) ## renderizando

def desenha_chao():
    global chaoSize,off_x,off_y,off_z
    # aplica a matriz model
    # rotacao
    angle = 0.0
    r_x = off_x; r_y = 0.0; r_z = 1.0    
    # translacao
    t_x = 100.0; t_y = off_y; t_z = off_z    
    # escala
    s_x = chaoSize; s_y =chaoSize; s_z = chaoSize    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 3)
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 4596, 102900-4596) ## renderizando

def desenha_Tree():
    # aplica a matriz model
    # rotacao
    global obj_x,obj_y,obj_z
    angle = 0.0
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = -101.5; t_y = -12.50; t_z = 43.0
    
    # escala
    s_x = 15.0; s_y = 15.0; s_z = 15.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo# desenha o modelo
    glBindTexture(GL_TEXTURE_2D, 5)
    glDrawArrays(GL_TRIANGLES, 102900, 139344-102900) 
    
    glBindTexture(GL_TEXTURE_2D, 6)
    glDrawArrays(GL_TRIANGLES, 139344, 195714-139344) 


def desenha_frozen():
    # aplica a matriz model
    # rotacao
    angle = -90.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = -25.0; t_y = -12.0; t_z = 152.5
    
    # escala
    s_x = 8.0; s_y = 8.0; s_z = 8.0
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 7)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 195714, 207645-195714) ## renderizando


obj_x=-34
obj_y=-11.5
obj_z=132.5

def desenha_dog():
    # aplica a matriz model
    # rotacao
    angle = -90.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = 15.0; t_y =-15.0; t_z = 194.0
    
    # escala
    s_x = .10; s_y = .10; s_z = .10
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    # desenha o modelo
    glBindTexture(GL_TEXTURE_2D, 8)
    glDrawArrays(GL_TRIANGLES, 207645, 221943-207645) 

#retorna a altura na posição x,z Tentar outra abordagem, esta está ruim
def getHei(x,z,vertList):
    global chaoSize, off_x,off_y,off_z
    tolerancia =2.
    #checa qual é o vertice mais proximo, quando achar basta retornar
    for v in vertList:
        vx = float(v[0])*chaoSize + off_x
        vz = float(v[2])*chaoSize + off_z
        #print(x,"-> ",float(v[0]))
        
        if math.isclose(x,vx,abs_tol=tolerancia):
            if math.isclose(z,vz,abs_tol=tolerancia):
                return float(v[1])*chaoSize+off_y
    print('não achou...')



jump = -21.50
def desenha_gollum():
    global jump
    # aplica a matriz model
    # rotacao
    angle=90
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = 6.0; t_y = jump; t_z = -244.0
    
    # escala
    s_x = 15.0; s_y = 15.0; s_z = 15.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 9)
    glDrawArrays(GL_TRIANGLES, 221943, 226224-221943) ## renderizando
    glBindTexture(GL_TEXTURE_2D, 10)
    glDrawArrays(GL_TRIANGLES, 226224, 227010-226224) ## renderizando
    glBindTexture(GL_TEXTURE_2D, 11)
    glDrawArrays(GL_TRIANGLES, 227010, 227823-227010) ## renderizando
    glBindTexture(GL_TEXTURE_2D, 12)
    glDrawArrays(GL_TRIANGLES, 227823, 231060-227823) ## renderizando

def desenha_ring():
    # aplica a matriz model
    # rotacao
    angle=90
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = -5.0; t_y = 110; t_z = -26.0
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 13)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 231060, 233364-231060) ## renderizando


cameraPos   = glm.vec3(-50.0,  110.0,  200.0)
cameraFront = glm.vec3(-25.0,  -12.0, 152.0)
cameraUp    = glm.vec3(0.0,  1.0,  0.0)

polygonal_mode = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp,polygonal_mode,obj_x,obj_y,obj_z
    """para limitar a camera, a idea era criar uma função (getHei) que retorne a coordenada Y (altura) do chao e do céu
    e limitar a movimentação da câmera. Porém desta forma fica muito ineficiente e mesmo assim (até a data de entrega)
    não consegui retornar o valor desejado
    """
    #print('camera Y:',cameraPos.y, ' Terrain heigth: ',getHei(cameraPos.x,cameraPos.z,chaoVert))

    #controles de objeto (para descobrir os valores de translação de cada matriz model)
    if key == 266 and (action==1 or action==2): # tecla pageup
        obj_z+=.5
        print('Z: ',obj_z)
    if key == 267 and (action==1 or action==2): # tecla pagedown
        obj_z-=.5
        print('Z: ',obj_z)
    if key == 268 and (action==1 or action==2): # tecla home
        obj_y+=.5
        print('Y: ',obj_y)
    if key == 269 and (action==1 or action==2): # tecla end
        obj_y-=.5
        print('Y: ',obj_y)
    if key == 260 and (action==1 or action==2): # tecla insert
        obj_x+=.5
        print('X: ',obj_x)
    if key == 261 and (action==1 or action==2): # tecla delete
        obj_x-=.5
        print('X: ',obj_x)

    cameraSpeed = 3.2
    #movimentar a camera um pouco mais rapido
    if key == 79 and (action==1 or action==2): # tecla O sobe rapidao
        cameraPos += 30 * cameraFront
    if key == 76 and (action==1 or action==2): # tecla l desce rapidao
        cameraPos -= 30 * cameraFront

    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
      
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True
        
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

#teste de scrool do mouse
def scroll_callback(window, xoffset, yoffset):    
    global cameraPos 
    cameraPos += yoffset
    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)
glfw.set_scroll_callback(window,scroll_callback)

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
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp)
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 3000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

glEnable(GL_DEPTH_TEST) ### importante para 3D
   
#fazer o gollum ficar pulando
jump_speed =0.35
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    
    

    desenha_casa()   
    desenha_cama()
    desenha_sky()
    desenha_chao()
    desenha_Tree()
    desenha_frozen()
    desenha_dog()
    desenha_gollum()
    desenha_ring()
    
    #gollum fica se mechendo...
    if jump>-14:
        jump_speed= -0.35
    if jump <-21.50:
        jump_speed = 0.35
    jump+=jump_speed

    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_FALSE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_FALSE, mat_projection)   
    

    
    glfw.swap_buffers(window)

glfw.terminate()
