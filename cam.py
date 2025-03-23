import cv2
import face_recognition
import numpy as np
import os

# Caminho da pasta de rostos
rostos_dir = "rostos"

# Listas para armazenar as codificações e os nomes
known_face_encodings = []
known_face_names = []

# Percorrer todas as pastas dentro de "rostos"
for pessoa in os.listdir(rostos_dir):
    pessoa_path = os.path.join(rostos_dir, pessoa)

    # Verifica se é um diretório (pasta)
    if os.path.isdir(pessoa_path):
        for imagem_nome in os.listdir(pessoa_path):
            imagem_path = os.path.join(pessoa_path, imagem_nome)

            # Carregar e codificar a imagem
            imagem = face_recognition.load_image_file(imagem_path)
            encodings = face_recognition.face_encodings(imagem)

            # Se houver um rosto na imagem, armazenar a codificação
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(pessoa)  # Nome da pasta vira o nome da pessoa

# Iniciar captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Capturar frame da webcam
    ret, frame = video_capture.read()

    # Converter para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostos e codificá-los
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Verificar se os rostos detectados correspondem aos conhecidos
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        # Usar a menor distância para melhorar a precisão
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Desenhar retângulo e nome na tela
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir o resultado
    cv2.imshow("Reconhecimento Facial", frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
