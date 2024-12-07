import cv2
import numpy as np
import time

# Caminhos dos arquivos
ARQUIVO_VIDEO = 'videos/nature.mp4'  # Substitua pelo caminho do seu vídeo ao vivo
ARQUIVO_MODELO = 'models/frozen_inference_graph.pb'  # Modelo treinado
ARQUIVO_CFG = 'models/ssd_mobilenet_v2_coco.pbtxt'  # Configuração do modelo
ARQUIVO_LABELS = 'models/coco_labels.txt'  # Arquivo com as classes

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    try:
        modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def carregar_classes(ARQUIVO_LABELS):
    try:
        with open(ARQUIVO_LABELS, 'r') as f:
            classes = [linha.strip() for linha in f.readlines()]
    except FileNotFoundError:
        print("Erro: Arquivo de classes não encontrado.")
        exit()
    return classes

def tirar_foto(frame, contador):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    nome_arquivo = f"fotos/animal_detectado_{contador}_{timestamp}.jpg"
    cv2.imwrite(nome_arquivo, frame)
    print(f"Foto salva: {nome_arquivo}")

def main():
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    modelo = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    classes = carregar_classes(ARQUIVO_LABELS)
    
    pausado = False
    contador_fotos = 0
    ultima_foto_tempo = 0  # Timestamp da última foto tirada

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob e execução do modelo
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            modelo.setInput(blob)
            deteccoes = modelo.forward()

            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:  # Apenas objetos com confiança acima de 50%
                    classe_id = int(deteccoes[0, 0, i, 1]) if i < deteccoes.shape[2] else 0
                    if classe_id < len(classes):
                        label = classes[classe_id]
                    else:
                        label = 'unknown'
                    # Verifica se é um animal (específico para classes relevantes)
                    if label in ['elephant', 'deer', 'bear', 'zebra', 'lion']:
                        (altura, largura) = frame.shape[:2]
                        caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
                        (inicioX, inicioY, fimX, fimY) = caixa.astype("int")

                        # Desenha a caixa e o rótulo no frame
                        cv2.rectangle(frame, (inicioX, inicioY), (fimX, fimY), (255, 0, 0), 2)
                        cv2.putText(frame, f"{label}: {int(confianca * 100)}%", (inicioX, inicioY - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                        # Verifica o tempo desde a última foto
                        tempo_atual = time.time()
                        if tempo_atual - ultima_foto_tempo >= 30:  # Verifica se 60 segundos se passaram
                            tirar_foto(frame, contador_fotos)
                            contador_fotos += 1
                            ultima_foto_tempo = tempo_atual

        # Exibe o vídeo com as detecções
        cv2.imshow("Detecção de Animais Selvagens", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):  # Sai do loop ao pressionar 'q'
            break
        elif tecla == ord('p'):  # Pausa ou continua ao pressionar 'p'
            pausado = not pausado

    # Liberação de recursos
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
