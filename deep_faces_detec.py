#import pip
#pip.main(["install","matplotlib"])



from mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cv2

def redimensiona(img):
    #função para redimensionando a imagem
    largura = img.shape[1]
    altura = img.shape[0]
    #calcula a proporção
    proporcao = float(altura/largura)
    #define uma nova largura em pixels
    largura_nova = 720
    altura_nova = int(largura_nova*proporcao)
    #define o novo tamanho
    novo_tamanho = (largura_nova,altura_nova)

    #redimensiona a imagem
    img_red = cv2.resize(img,novo_tamanho, interpolation=cv2.INTER_AREA)
    return img_red

def draw_rectangle(imagem, faces):
    ax = plt.gca()

    for face in faces:
        x, y, width, height = face['box']
        rect = Rectangle((x,y), width, height, color='yellow', fill=False)
        ax.add_patch(rect)
    plt.imshow(imagem)
    plt.show()



def face_detect(imagem):
    #variável que armazena a cor da seleção
    BLUE_COLOR = (255,255,0)

    #cria o detector
    detector = MTCNN()

    #origem da imagem
    image_path = imagem
    
    #faz a leitura da imagem
    #img = cv2.imread(image_path)
    img = plt.imread(image_path)

    #converte para escala RGB
    #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #realiza a detecção das faces
    faces = detector.detect_faces(img)    

    #define o tamanho da janela
    cv2.WINDOW_FULLSCREEN 

    #Redimensionando a imagem
    img_red = img

    
    #Escrevendo na tela
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_red, str(len(faces)) + 'faces', (5,56), fonte,2,BLUE_COLOR,2,cv2.LINE_AA)
    

    #Exibe a imagem já redimensionada
    #cv2.imshow('Detecção de faces', img_red)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    draw_rectangle(img, faces)




################################
################################
#Chamada da função principal
face_detect('imagens/im (1).jpg')