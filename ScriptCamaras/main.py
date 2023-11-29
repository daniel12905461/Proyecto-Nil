import cv2

# Cargar el clasificador de cascada para la detección de rostros
faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Configurar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Variable para seguir el estado de detección
rostro_detectado = False

while True:
    # Capturar cuadro de video desde la cámara
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el cuadro
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Verificar si se detectó un rostro y no se ha enviado el mensaje
    if len(faces) > 0 and not rostro_detectado:
        # Imprimir un mensaje por consola
        print("¡Rostro detectado!")

        # Cambiar el estado de detección para que no se imprima el mensaje nuevamente
        rostro_detectado = True
    elif len(faces) == 0:
        # Reiniciar el estado de detección si no se detectan rostros
        rostro_detectado = False

    # Iterar sobre cada rostro detectado
    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el cuadro resultante en una ventana llamada 'frame'
    cv2.imshow('frame', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar ventanas al salir
cap.release()
cv2.destroyAllWindows()
