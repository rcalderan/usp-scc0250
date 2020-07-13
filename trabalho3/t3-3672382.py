"""
USP  Computação Gráfica - SCC0250 2020/1
Prof. Ricardo Marcondes Marcacini

Richard Calderan - 3672382

Trabalho 3 

Requisitos:
1. O cenário deve conter um ambiente interno e externo. O ambiente interno deve
conter pelo menos três modelos. O ambiente externo deve conter pelo menos três
modelos. Pelo menos 1 modelo deve ser animado.
ok
2. Adicionar uma fonte de luz no ambiente externo. A fonte de luz pode ser um sol, lua
ou qualquer objeto.
ok
3. A fonte de luz do ambiente externo deve se movimentar ao longo do cenário. O
movimento da fonte de luz do ambiente externo deve impactar a iluminação dos
modelos.
ok (sol)
4. Cada modelo no ambiente externo deve ter seus próprios parâmetros de iluminação,
ka (ambiente), kd (difusa) e ks (especular).
ok
5. Adicionar uma fonte de luz no ambiente interno. A fonte de luz é fixa, podendo ser
uma lâmpada, um abajur, etc.
ok (lâmpada)
6. Faça com que os modelos no ambiente interno sejam mais afetados pela luz do
ambiente interno do que pela luz do ambiente externo.
ok
7. Cada modelo no ambiente interno deve ter seus próprios parâmetros de iluminação,
ka (ambiente), kd (difusa) e ks (especular).
ok
8. Faça com que a tecla L desligue e ligue a luz do ambiente interno, tornando o
ambiente mais escuro, sendo iluminado apenas por uma luz ambiente fraca.
ok
9. Faça com que as teclas U e P aumentem e reduzam a intensidade da luz ambiente.
ok
10. Escolha um dos seus modelos para aplicar um efeito (bem visível) de reflexão
especular.
ok Anel. É melhor visualizado com a luz externa, posicionando a câmera do lado de fora de frente pra porta. 


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
window = glfw.create_window(largura, altura, "Trabalho 3 - Iluminação", None, None)
glfw.make_context_current(window)

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;
        
       
        varying vec2 out_texture;
        varying vec3 out_fragPos;
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(model * vec4(position, 1.0));
            out_normal = normals;            
        }
        """

fragment_code = """

        // parametros da iluminacao ambiente e difusa
        uniform vec3 lightPos; // define coordenadas de posicao da luz
        uniform float ka; // coeficiente de reflexao ambiente
        uniform float kd; // coeficiente de reflexao difusa    
        
        // parametros da iluminacao especular
        uniform vec3 viewPos; // define coordenadas com a posicao da camera/observador
        uniform float ks; // coeficiente de reflexao especular
        uniform float ns; // expoente de reflexao especular
        
        // parametro com a cor da(s) fonte(s) de iluminacao
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 i_lightColor = vec3(1.0, 1.0, 0.0);

        // parametros recebidos do vertex shader
        varying vec2 out_texture; // recebido do vertex shader
        varying vec3 out_normal; // recebido do vertex shader
        varying vec3 out_fragPos; // recebido do vertex shader
        uniform sampler2D samplerTexture;        
        
        
        void main(){
        
            // calculando reflexao ambiente
            vec3 ambient = ka * lightColor ;    
            //

            // calculando reflexao difusa
            vec3 norm = normalize(out_normal); // normaliza vetores perpendiculares

            vec3 lightDir = normalize(lightPos - out_fragPos); // direcao da luz
            float diff = max(dot(norm, lightDir), 0.0); // verifica limite angular (entre 0 e 90)
            vec3 diffuse = kd * diff * lightColor; // iluminacao difusa
            
            // calculando reflexao especular
            vec3 viewDir = normalize(viewPos - out_fragPos); // direcao do observador/camera
            vec3 reflectDir = reflect(-lightDir, norm); // direcao da reflexao
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), ns);
            vec3 specular = ks * spec  * lightColor; 

            // aplicando o modelo de iluminacao
            vec4 texture = texture2D(samplerTexture, out_texture);
            vec4 result = vec4((ambient+ diffuse + specular),1.0) *  texture; // aplica iluminacao
            
            gl_FragColor = result;       
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

modelo = load_model_from_file('assets/casa/casa.obj')
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
load_texture_from_file(0,'assets/casa/casa.jpg')

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
#load_texture_from_file(4,'assets/chao/field.jpg')

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


modelo = load_model_from_file('assets/sun/esfera.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo luz externa. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo luz externa. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(14,'assets/sun/luz.png')


modelo = load_model_from_file('assets/sun/esfera.obj')
### inserindo vertices do modelo no vetor de vertices
print('Processando modelo luz interna. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
print('Processando modelo luz interna. Vertice final:',len(vertices_list))
### inserindo coordenadas de textura do modelo no vetor de texturas
### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(14,'assets/sun/luz.png')


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

#globais
ka_offset=0
ka = 0.3 # coeficiente de reflexao ambiente do modelo
isInside=False
turnOn = True

def desenha_casa():
    global isInside, turnOn,ka_offset
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
    

    #### define parametros de ilumincao do modelo
    if isInside:
        ka = 0.3+ka_offset # coeficiente de reflexao ambiente do modelo
        kd = 0.8 # coeficiente de reflexao difusa do modelo
        if not turnOn:
            kd = 0.0 
    else:
        ka= .1+ka_offset
        kd= .6 
    ks = 0.0005 # coeficiente de reflexao especular do modelo
    ns = 1024.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu

    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)
    
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 0, 1770) ## renderizando

def desenha_cama():
    global isInside,turnOn,ka_offset

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
       
    
    #### define parametros de ilumincao do modelo
    ka = ka_offset+0.3 # coeficiente de reflexao ambiente do modelo
    if isInside:        
        kd = 0.5 # coeficiente de reflexao difusa do modelo        
        if not turnOn:
            kd = 0.0 
    else:
        kd=.001
    ks = 0.05 # coeficiente de reflexao especular do modelo
    ns = 0.001 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu   
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ks pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu  
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 1)
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 1770, 3156-1770) ## renderizando


def desenha_sky():
    global ka_offset
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
    
    
    #### define parametros de ilumincao do modelo
    ka = ka_offset+0.1 # coeficiente de reflexao ambiente do modelo
    kd = 0.5 # coeficiente de reflexao difusa do modelo
    ks = 0.1 # coeficiente de reflexao especular do modelo
    ns = 0.01 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu     
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu         
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu
 
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 3156, 4596-3156) ## renderizando

def desenha_chao():
    global ka_offset
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
    
    #### define parametros de ilumincao do modelo
    ka = ka_offset+0.1 # coeficiente de reflexao ambiente do modelo
    kd = 0.6 # coeficiente de reflexao difusa do modelo
    ks = 0.001 # coeficiente de reflexao especular do modelo
    ns = 1024.0 # expoente de reflexao especular

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu  
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu


    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 3)
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 4596, 102900-4596) ## renderizando

def desenha_Tree():
    # aplica a matriz model
    # rotacao
    global ka_offset
    angle = 0.0
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = -101.5; t_y = -12.50; t_z = 43.0
    
    # escala
    s_x = 15.0; s_y = 15.0; s_z = 15.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)

    
    #### define parametros de ilumincao do modelo
    ka = 0.1 +ka_offset # coeficiente de reflexao ambiente do modelo
    kd = 0.61 # coeficiente de reflexao difusa do modelo
    ks = 0.001 # coeficiente de reflexao especular do modelo
    ns = 1024.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu   
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu

    #define id da textura do modelo# desenha o modelo
    glBindTexture(GL_TEXTURE_2D, 5)
    glDrawArrays(GL_TRIANGLES, 102900, 139344-102900) 
    
    glBindTexture(GL_TEXTURE_2D, 6)
    glDrawArrays(GL_TRIANGLES, 139344, 195714-139344) 

def desenha_frozen():
    global ka_offset,isInside,turnOn
    # aplica a matriz model
    # rotacao
    angle = -90.0
    r_x = 0.0; r_y = 1.0; r_z = 0.0
    
    # translacao
    t_x = -25.0; t_y = -12.0; t_z = 132.5
    
    # escala
    s_x = 8.0; s_y = 8.0; s_z = 8.0
    
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
    
    ka = 0.3+ka_offset # coeficiente de reflexao ambiente do modelo
    if isInside:
        kd = 0.9 # coeficiente de reflexao difusa do modelo        
        if not turnOn:
            kd = 0.0 
    else:
        kd = 0.001 # coeficiente de reflexao difusa do modelo
    ks = .001 # coeficiente de reflexao especular do modelo
    ns = 512 # expoente de reflexao especular
    
    #i_kd= 0.001

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu   
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu          
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu

    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 7)
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 195714, 207645-195714) ## renderizando


obj_x=23
obj_y=168
obj_z=-41
#acho q deu certo. Agora é só ajustar os coeficientes
# sim. o modelo do anel não parece estar mostrando o specular. mas ahco q
def desenha_dog():
    global ka_offset
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
    
    #### define parametros de ilumincao do modelo
    ka = 0.12+ka_offset # coeficiente de reflexao ambiente do modelo
    kd = 0.62 # coeficiente de reflexao difusa do modelo
    ks = 0.01 # coeficiente de reflexao especular do modelo
    ns = 512 # expoente de reflexao especular

    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    loc_ka = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ka, ks) ### envia ns pra gpu    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu

    #define id da textura do modelo
    # desenha o modelo
    glBindTexture(GL_TEXTURE_2D, 8)
    glDrawArrays(GL_TRIANGLES, 207645, 221943-207645) 

 
jump = -21.50
def desenha_gollum():
    global jump,ka_offset
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

    
    #### define parametros de ilumincao do modelo
    ka = 0.13 +ka_offset# coeficiente de reflexao ambiente do modelo
    kd = 0.57 # coeficiente de reflexao difusa do modelo
    ks = 0.01 # coeficiente de reflexao especular do modelo
    ns = 512 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu     
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu

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
    global turnOn
    # aplica a matriz model
    # rotacao
    angle=90
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = -5.0; t_y = 155; t_z = -26.0
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)
       
    #define id da textura do modelo
    glBindTexture(GL_TEXTURE_2D, 13)
    
    ka = 0.3 +ka_offset# coeficiente de reflexao ambiente do modelo
    if isInside:
        kd = 0.8 # coeficiente de reflexao difusa do modelo        
        if not turnOn:
            kd = 0.0 
    else:
        kd = 0.001 # coeficiente de reflexao difusa do modelo
    ks = 1.0 # coeficiente de reflexao especular do modelo
    ns = 2.0 # expoente de reflexao especular
    #acho que é só entender direito como funciona isso kkk
    #

    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    loc_ka = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, kd) ### envia ka pra gpu     
    loc_ka = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka,ks) ### envia ka pra gpu   
    loc_ka = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ns) ### envia ka pra gpu
    
    # desenha o modelo
    glDrawArrays(GL_TRIANGLES, 231060, 233364-231060) ## renderizando


cameraPos   = glm.vec3(-50.0,  110.0,  200.0)
cameraFront = glm.vec3(-25.0,  -12.0, 152.0)
cameraUp    = glm.vec3(0.0,  1.0,  0.0)

def desenha_sun(ang):
    global isInside
    # aplica a matriz model
    # rotacao
    angle=90
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    raio=1450
    t_x = np.cos(ang)*raio; t_y = 0; t_z =  np.sin(ang)*raio
    
    # escala
    s_x = 30.0; s_y = 30.0; s_z = 30.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)


    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular
    
    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu    
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu        
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu            
    
    if not isInside:
        loc_light_pos = glGetUniformLocation(program, "lightPos") # recuperando localizacao da variavel lightPos na GPU
        glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz


    glBindTexture(GL_TEXTURE_2D, 14)    
    glDrawArrays(GL_TRIANGLES, 233364, 236244-233364) ## renderizando

def desenha_lampada():
    global isInside
    # aplica a matriz model
    # rotacao
    angle=90
    r_x = 0.0; r_y = 0.0; r_z = 1.0
    
    # translacao
    t_x = 19; t_y = 192; t_z = 0
    
    # escala
    s_x = 1.0; s_y = 1.0; s_z = 1.0
    mat_model = model(angle, r_x, r_y, r_z, t_x, t_y, t_z, s_x, s_y, s_z)
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_TRUE, mat_model)


    #### define parametros de ilumincao do modelo
    ka = 1 # coeficiente de reflexao ambiente do modelo
    kd = 1 # coeficiente de reflexao difusa do modelo
    ks = 1 # coeficiente de reflexao especular do modelo
    ns = 1000.0 # expoente de reflexao especular


    loc_ka = glGetUniformLocation(program, "ka") # recuperando localizacao da variavel ka na GPU
    glUniform1f(loc_ka, ka) ### envia ka pra gpu
    
    loc_kd = glGetUniformLocation(program, "kd") # recuperando localizacao da variavel kd na GPU
    glUniform1f(loc_kd, kd) ### envia kd pra gpu   
    
    loc_ks = glGetUniformLocation(program, "ks") # recuperando localizacao da variavel ks na GPU
    glUniform1f(loc_ks, ks) ### envia ns pra gpu         
    
    loc_ns = glGetUniformLocation(program, "ns") # recuperando localizacao da variavel ns na GPU
    glUniform1f(loc_ns, ns) ### envia ns pra gpu           
    
    if isInside:        
        loc_light_pos = glGetUniformLocation(program, "lightPos") # recuperando localizacao da variavel lightPos na GPU
        glUniform3f(loc_light_pos, t_x, t_y, t_z) ### posicao da fonte de luz

    glBindTexture(GL_TEXTURE_2D, 14)    
    glDrawArrays(GL_TRIANGLES, 236244, 239124-236244) ## renderizando


#check if the observer is inside house
def checkPosition(cameraPos):
    global isInside
    if cameraPos[0]>=-195 and cameraPos[0]<= 107 and cameraPos[1]>=-10 and cameraPos[1]<= 19 and  cameraPos[2]>=-52 and cameraPos[2]<= -1.5:
        isInside= True
    else:
        isInside=False
def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp,turnOn,ka_offset,obj_x,obj_y,obj_z
    
    #para ajuste de posições
    if key == 266 and (action==1 or action==2): # tecla pageup
        obj_y+=.5
        print('Y: ',obj_y)
    if key == 267 and (action==1 or action==2): # tecla pagedown        
        obj_y-=.5
        print('Y: ',obj_y)
    if key == 263 and (action==1 or action==2): # tecla up
        obj_z+=.5
        print('Z: ',obj_z)
    if key == 262 and (action==1 or action==2): # tecla down
        obj_z-=.5
        print('Z: ',obj_z)
    if key == 265 and (action==1 or action==2): # tecla ri
        obj_x+=.5
        print('X: ',obj_x)
    if key == 264 and (action==1 or action==2): # tecla le
        obj_x-=.5
        print('X: ',obj_x)

    cameraSpeed = 3.2

    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
        #print(cameraPos)
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
        #print(cameraPos)
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        #print(cameraPos)
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        #print(cameraPos)
    

    if key == 76 and (action==1 or action==2): # tecla l 
        turnOn=not turnOn
    
    if key == 85 and (action==1 or action==2): # tecla u
        ka_offset+=0.05
        print("ambient (+)")
    if key == 80 and (action==1 or action==2): # tecla p
        ka_offset-=0.05
        print("ambient (-)")
    checkPosition(cameraPos)
        
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
ang=0
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(1.0, 1.0, 1.0, 1.0)

    desenha_casa()   
    desenha_cama()
    desenha_sky()
    desenha_chao()
    desenha_Tree()
    desenha_frozen()
    desenha_dog()
    desenha_gollum()
    desenha_ring()

    #movimentando o sol
    ang+=.02
    desenha_sun(ang)
    desenha_lampada()
    
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
