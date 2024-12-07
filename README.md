# Projeto de Visão Computacional para Detecção de Animais Selvagens

Projeto desenvolvido para capturar imagens de animais selvagens em ambientes ao ar livre, utilizando visão computacional e modelos de detecção de objetos.

## Objetivo do Projeto

O objetivo deste projeto é realizar a detecção de animais selvagens em vídeos ao vivo ou gravações de câmeras de vigilância ao ar livre. O sistema utiliza um modelo de deep learning (SSD MobileNet) para identificar animais específicos (como elefantes, cervos, ursos, zebras, e leões) e, ao detectar um animal, uma foto é capturada e salva automaticamente. O código foi ajustado para tirar apenas uma foto por minuto, evitando capturas repetidas de animais em sequência.

## Configuração do Ambiente Virtual

### Passos para criar e ativar um ambiente virtual:

1. **Criar o ambiente virtual:**

   ```bash
   python -m venv env-visao
   ```

2. **Ativar o ambiente virtual:**

   No macOS e Linux:

   ```bash
   source ./env-visao/bin/activate
   ```

   No Windows:

   ```bash
   .\env-visao\Scripts\activate
   ```

## Instalação de Dependências

Após ativar o ambiente virtual, instale as dependências necessárias para o projeto com o seguinte comando:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.0.0
opencv-python==4.10.0.84
```

## Estrutura de Pastas

O projeto possui a seguinte estrutura:

```
/projeto
    /fotos                # Onde as fotos dos animais detectados são salvas
    /models               # Contém os arquivos do modelo de detecção e configuração
    /videos               # Vídeos para detecção
    main.py               # Arquivo principal para execução do código
    requirements.txt      # Dependências do projeto
```

## Como Utilizar

1. **Modelo e Configuração:** O projeto usa o modelo SSD MobileNet V2 treinado com o conjunto de dados COCO. Certifique-se de que o arquivo `frozen_inference_graph.pb`, `ssd_mobilenet_v2_coco.pbtxt`, e `coco_labels.txt` estejam corretamente configurados.

   **Modelo:** Baixe o modelo clicando [aqui](https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz), copie o arquivo `frozen_inference_graph.pb` para a pasta `/models`.
   
2. **Executar o Código:** Para iniciar a detecção, execute o código no arquivo `main.py`. O sistema irá capturar uma foto dos animais detectados a cada 60 segundos.

   ```bash
   python main.py
   ```

   Certifique-se de substituir o caminho do vídeo no arquivo pelo vídeo que você deseja monitorar.

3. **Verifique as Fotos:** As fotos serão salvas na pasta `/fotos` com o nome do arquivo incluindo a data e hora da captura.

## Verificação da Instalação

Para verificar se as bibliotecas foram instaladas corretamente, você pode executar o seguinte código Python:

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Desativação do Ambiente Virtual

Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual com o comando:

```bash
deactivate
```
