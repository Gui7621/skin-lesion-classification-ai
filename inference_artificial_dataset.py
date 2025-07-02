import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """
    Carrega o modelo de classificação de melanoma
    """
    print(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Modelo carregado com sucesso!")
    return model

def load_images_from_folder(folder_path, target_size=(224, 224)):
    """
    Carrega todas as imagens de uma pasta e redimensiona para o tamanho alvo
    """
    images = []
    filenames = []
    
    # Extensões de imagem suportadas
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    print(f"Carregando imagens de: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            
            # Carrega a imagem usando OpenCV
            img = cv2.imread(img_path)
            
            if img is not None:
                # Converte de BGR para RGB (OpenCV usa BGR por padrão)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensiona para 224x224
                img_resized = cv2.resize(img_rgb, target_size)
                
                images.append(img_resized)
                filenames.append(filename)
            else:
                print(f"Erro ao carregar imagem: {filename}")
    
    print(f"Carregadas {len(images)} imagens de {folder_path}")
    return np.array(images), filenames

def predict_images(model, images):
    """
    Faz predições para um conjunto de imagens
    """
    print("Fazendo predições...")
    
    # Converte para float32 (sem normalização conforme solicitado)
    images_float = images.astype(np.float32)
    
    # Faz as predições
    predictions = model.predict(images_float)
    
    # Como é sigmoid, as predições já estão entre 0 e 1
    return predictions.flatten()

def classify_predictions(predictions, threshold=0.5):
    """
    Classifica as predições baseado no threshold
    """
    return ["mel" if pred > threshold else "nv" for pred in predictions]

def print_results(filenames, predictions, predicted_classes, true_class):
    """
    Imprime os resultados das predições
    """
    print(f"\n=== Resultados para classe {true_class.upper()} ===")
    print("Arquivo\t\t\tClasse Prevista\tScore")
    print("-" * 60)
    
    for filename, pred_class, score in zip(filenames, predicted_classes, predictions):
        print(f"{filename:<25}\t{pred_class}\t\t{score:.4f}")

def plot_confusion_matrix(y_true, y_pred, class_names=['nv', 'mel']):
    """
    Plota a matriz de confusão
    """
    # Calcula a matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Cria o gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - Classificação de Melanoma')
    plt.xlabel('Classe Prevista')
    plt.ylabel('Classe Verdadeira')
    plt.tight_layout()
    plt.show()
    
    # Calcula e imprime métricas
    print("\n=== Métricas de Performance ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calcula acurácia manualmente
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")

def main():
    # Configurações
    model_path = "melanoma_classifier_fixed.h5"
    nv_folder = "skin_cancer_artificial_dataset/nv"  # Pasta com imagens benignas
    mel_folder = "skin_cancer_artificial_dataset/mel"  # Pasta com imagens malignas
    
    # Verifica se as pastas existem
    if not os.path.exists(nv_folder):
        print(f"Erro: Pasta '{nv_folder}' não encontrada!")
        return
    
    if not os.path.exists(mel_folder):
        print(f"Erro: Pasta '{mel_folder}' não encontrada!")
        return
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo '{model_path}' não encontrado!")
        return
    
    try:
        # 1. Carrega o modelo
        model = load_model(model_path)
        
        # 2 e 3. Carrega e redimensiona as imagens
        nv_images, nv_filenames = load_images_from_folder(nv_folder)
        mel_images, mel_filenames = load_images_from_folder(mel_folder)
        
        if len(nv_images) == 0 and len(mel_images) == 0:
            print("Nenhuma imagem encontrada nas pastas!")
            return
        
        # 4. Faz predições para cada conjunto
        results = []
        all_true_labels = []
        all_pred_labels = []
        all_filenames = []
        all_scores = []
        
        if len(nv_images) > 0:
            nv_predictions = predict_images(model, nv_images)
            nv_pred_classes = classify_predictions(nv_predictions)
            
            # 5. Imprime resultados para imagens nv
            print_results(nv_filenames, nv_predictions, nv_pred_classes, "nv")
            
            # Armazena para matriz de confusão
            all_true_labels.extend(["nv"] * len(nv_filenames))
            all_pred_labels.extend(nv_pred_classes)
            all_filenames.extend(nv_filenames)
            all_scores.extend(nv_predictions)
        
        if len(mel_images) > 0:
            mel_predictions = predict_images(model, mel_images)
            mel_pred_classes = classify_predictions(mel_predictions)
            
            # 5. Imprime resultados para imagens mel
            print_results(mel_filenames, mel_predictions, mel_pred_classes, "mel")
            
            # Armazena para matriz de confusão
            all_true_labels.extend(["mel"] * len(mel_filenames))
            all_pred_labels.extend(mel_pred_classes)
            all_filenames.extend(mel_filenames)
            all_scores.extend(mel_predictions)
        
        # 6. Gera matriz de confusão
        if len(all_true_labels) > 0:
            plot_confusion_matrix(all_true_labels, all_pred_labels)
            
            # Resumo geral
            print(f"\n=== RESUMO GERAL ===")
            print(f"Total de imagens processadas: {len(all_filenames)}")
            print(f"Imagens nv (benigno): {len(nv_filenames) if len(nv_images) > 0 else 0}")
            print(f"Imagens mel (maligno): {len(mel_filenames) if len(mel_images) > 0 else 0}")
            
            # Estatísticas dos scores
            print(f"\nEstatísticas dos scores:")
            print(f"Score mínimo: {min(all_scores):.4f}")
            print(f"Score máximo: {max(all_scores):.4f}")
            print(f"Score médio: {np.mean(all_scores):.4f}")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()