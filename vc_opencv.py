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



def face_detect(imagem):

    #variável que armazena a cor da seleção
    BLUE_COLOR = (255,0,0)

    #origem da imagem
    image_path = imagem
    #carrega o modelo
    modelo = 'haarcascade_frontalface_default.xml'
    #cria um classficador
    clf = cv2.CascadeClassifier(modelo)
    n_faces = 0

    #faz a leitura da imagem
    img = cv2.imread(image_path)
    #converte para escala cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #realiza a detecção das faces na imagem
    faces = clf.detectMultiScale(gray, 1.3, 3)
    
    #desenhamos um retângulo utilizando as coordenadas 
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0),2)
        n_faces +=1
    #define o tamanho da janela
    cv2.WINDOW_FULLSCREEN
    
    #Redimensionando a imagem
    img_red = redimensiona(img)

    #Mostra um retângulo no canto superior esquerdo
    (b, g, r) = img_red[200, 200]
    img_red[198:202, 198:202] = (0, 0, 255)
    img_red[10:80, 10:380] = (b, g, r)

    #Escrevendo na tela
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_red, str(n_faces) + 'faces', (5,56), fonte,2,BLUE_COLOR,2,cv2.LINE_AA)

    #Exibe a imagem já redimensionada
    cv2.imshow('Detecção de faces', img_red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



################################
################################
#Chamada da função principal
face_detect('imagens/im (1).jpg')



