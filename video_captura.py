import cv2

def capturar():

    #variável que armazena a cor da seleção
    BLUE_COLOR = (255,0,0)
    STROKE = 2

    ##Variável que recebe o método de captura na webcam padrão (ID 0)
    cam = cv2.VideoCapture(0)

    #carrega o modelo
    modelo = 'haarcascade_frontalface_default.xml'

    #cria um classficador
    clf = cv2.CascadeClassifier(modelo)

    #variável que receberá a contagem das faces
    n_faces = 0

    ##Laço de repetição para manter a captura ativa 
    while True:
        #variável que recebe a captura do frame
        ret, frame = cam.read()

        #Converte o frame para tons de cinza
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detectar as faces
        faces = clf.detectMultiScale(im_gray)

        #desenhamos um retângulo utilizando as coordenadas 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (BLUE_COLOR),2)
            n_faces +=1

        #Mostra um retângulo no canto superior esquerdo
        #(b, g, r) = frame[200, 200]
        #frame[198:202, 198:202] = (0, 0, 255)
        #frame[10:90, 10:90] = (b, g, r)

        #Escrevendo texto na tela
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(faces) , (5,56), fonte,2,BLUE_COLOR,2,cv2.LINE_AA)

        #comando para exibir a imagem
        cv2.imshow("Pressione ESC para sair", frame)
        
        
        #variável que armazena o valor da tecla pressionada    
        key = cv2.waitKey(1)
        #Se pressionar ESC interrompe
        if key%256 == 27:
            break
    #finaliza a webcam
    cam.release()
    #libera memória e finaliza janelas
    cv2.destroyAllWindows()


capturar()