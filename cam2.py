import cv2
from deepface import DeepFace
import os

# Caminho para os arquivos de Haar Cascade
haarcascade_path = 'C:/Users/Desktop/Documents/GitHub/MinhaPagina/haarcascade_frontalface_default.xml'  # Atualize o caminho

# Verifica se o arquivo Haar Cascade existe
if not os.path.exists(haarcascade_path):
    raise FileNotFoundError(f"Arquivo Haar Cascade não encontrado: {haarcascade_path}")

# Carregar o classificador de faces
face_cascade = cv2.CascadeClassifier(haarcascade_path)

# Caminho da pasta de rostos
rostos_dir = 'rostos'

# Iniciar captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame da webcam
    ret, frame = video_capture.read()

    # Converter a imagem para escala de cinza (necessário para o Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame usando o Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Se houver rostos detectados, proceder com o reconhecimento
    if len(faces) > 0:
        # Para cada rosto detectado, realizar a comparação com a base de dados
        for (x, y, w, h) in faces:
            # Desenhar um retângulo em volta do rosto
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Cortar a região do rosto
            face = frame[y:y+h, x:x+w]

            try:
                # Realizar a busca no banco de dados usando DeepFace
                result = DeepFace.find(face, db_path='C:/Users/Desktop/Documents/GitHub/MinhaPagina/rostos', enforce_detection=False)
                
                # Mostrar o nome da pessoa na tela
                if result:
                    name = result[0]['identity'][0]  # O nome estará no primeiro elemento
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                print(f"Erro ao tentar reconhecer o rosto: {e}")
                cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Exibir o resultado
    cv2.imshow("Reconhecimento Facial", frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
