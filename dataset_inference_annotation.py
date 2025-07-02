import os
import cv2

# Caminhos das pastas
pasta_nevus = "/mnt/c/skin_cancer_artificial_dataset/nv"
pasta_melanoma = "/mnt/c/skin_cancer_artificial_dataset/mel"

# Função para redimensionar e renomear imagens
def processar_imagens(pasta, prefixo, largura=600, altura=450):
    arquivos = sorted(os.listdir(pasta))
    for i, nome_atual in enumerate(arquivos, start=1):
        caminho_antigo = os.path.join(pasta, nome_atual)
        novo_nome = f"{prefixo}{i:02d}.jpeg"
        caminho_novo = os.path.join(pasta, novo_nome)

        # Leitura e redimensionamento com OpenCV
        imagem = cv2.imread(caminho_antigo)
        if imagem is None:
            print(f"Erro ao ler {nome_atual}")
            continue

        imagem_redimensionada = cv2.resize(imagem, (largura, altura))
        cv2.imwrite(caminho_novo, imagem_redimensionada)

        # Remove a imagem original se tiver nome diferente
        if nome_atual != novo_nome:
            os.remove(caminho_antigo)

# Processar ambas as classes
processar_imagens(pasta_nevus, 'nv')
processar_imagens(pasta_melanoma, 'mel')

