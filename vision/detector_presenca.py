"""
Módulo de Visão Computacional – Detecção de Presença.

Detecta se há uma pessoa próxima ao totem usando duas abordagens:
1. HOG (Histogram of Oriented Gradients) Person Detector — OpenCV
2. MediaPipe Face Detection — detecção facial como proxy de presença

O módulo é usado de duas formas:
- Em tempo real: integrado ao app do totem (Streamlit) como trigger de sessão
- Offline: processamento de imagens do dataset para treinar o classificador

Substitui funcionalmente o sensor PIR do ESP32 descrito na Sprint 1,
usando visão computacional em vez de sensor infravermelho.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# Tentar importar MediaPipe (fallback para HOG se não disponível)
try:
    import mediapipe as mp
    MEDIAPIPE_DISPONIVEL = True
except ImportError:
    MEDIAPIPE_DISPONIVEL = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DetectorPresenca:
    """
    Detector de presença humana em imagens.

    Combina dois métodos para maior robustez:
    - HOG: detecta corpo inteiro (silhuetas)
    - MediaPipe: detecta rostos (proxy para presença próxima)

    A decisão final usa votação: se qualquer método detecta presença,
    o resultado é positivo (OR lógico), priorizando recall alto —
    no contexto do totem, é melhor ativar a tela desnecessariamente
    do que ignorar um visitante real.
    """

    def __init__(self, metodo: str = "combinado"):
        """
        Inicializa o detector.

        Args:
            metodo: 'hog', 'mediapipe' ou 'combinado' (padrão)
        """
        self.metodo = metodo
        self.hog = None
        self.mp_face = None

        # Inicializar HOG
        if metodo in ("hog", "combinado"):
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Inicializar MediaPipe
        if metodo in ("mediapipe", "combinado") and MEDIAPIPE_DISPONIVEL:
            self.mp_face = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.4,
                model_selection=0,  # 0 = curta distância (< 2m), ideal para totem
            )

    def detectar_hog(self, imagem: np.ndarray) -> dict:
        """Detecta pessoas usando HOG + SVM."""
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Detectar pessoas
        rects, pesos = self.hog.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
        )

        deteccoes = []
        for (x, y, w, h), peso in zip(rects, pesos):
            deteccoes.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confianca": round(float(peso), 4),
            })

        return {
            "presenca": len(deteccoes) > 0,
            "num_pessoas": len(deteccoes),
            "confianca_max": max((d["confianca"] for d in deteccoes), default=0.0),
            "deteccoes": deteccoes,
            "metodo": "hog",
        }

    def detectar_mediapipe(self, imagem: np.ndarray) -> dict:
        """Detecta rostos usando MediaPipe Face Detection."""
        if self.mp_face is None:
            return {
                "presenca": False,
                "num_pessoas": 0,
                "confianca_max": 0.0,
                "deteccoes": [],
                "metodo": "mediapipe",
                "erro": "MediaPipe não disponível",
            }

        rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        resultado = self.mp_face.process(rgb)

        deteccoes = []
        if resultado.detections:
            h_img, w_img = imagem.shape[:2]
            for det in resultado.detections:
                bbox = det.location_data.relative_bounding_box
                deteccoes.append({
                    "bbox": [
                        int(bbox.xmin * w_img),
                        int(bbox.ymin * h_img),
                        int(bbox.width * w_img),
                        int(bbox.height * h_img),
                    ],
                    "confianca": round(det.score[0], 4),
                })

        return {
            "presenca": len(deteccoes) > 0,
            "num_pessoas": len(deteccoes),
            "confianca_max": max((d["confianca"] for d in deteccoes), default=0.0),
            "deteccoes": deteccoes,
            "metodo": "mediapipe",
        }

    def detectar(self, imagem: np.ndarray) -> dict:
        """
        Detecta presença humana na imagem.

        Args:
            imagem: array numpy BGR (OpenCV format)

        Returns:
            dict com presença detectada, confiança e detalhes
        """
        if self.metodo == "hog":
            return self.detectar_hog(imagem)
        elif self.metodo == "mediapipe":
            return self.detectar_mediapipe(imagem)
        else:
            # Método combinado: roda ambos e faz OR lógico
            res_hog = self.detectar_hog(imagem)
            res_mp = self.detectar_mediapipe(imagem)

            presenca = res_hog["presenca"] or res_mp["presenca"]
            confianca = max(res_hog["confianca_max"], res_mp["confianca_max"])
            num_pessoas = max(res_hog["num_pessoas"], res_mp["num_pessoas"])

            return {
                "presenca": presenca,
                "num_pessoas": num_pessoas,
                "confianca_max": round(confianca, 4),
                "metodo": "combinado",
                "detalhe_hog": res_hog,
                "detalhe_mediapipe": res_mp,
            }

    def detectar_arquivo(self, caminho: str) -> dict:
        """Detecta presença a partir de um arquivo de imagem."""
        if not os.path.exists(caminho):
            return {"erro": f"Arquivo não encontrado: {caminho}"}

        imagem = cv2.imread(caminho)
        if imagem is None:
            return {"erro": f"Não foi possível ler a imagem: {caminho}"}

        resultado = self.detectar(imagem)
        resultado["arquivo"] = caminho
        resultado["resolucao"] = f"{imagem.shape[1]}x{imagem.shape[0]}"
        resultado["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return resultado

    def anotar_imagem(self, imagem: np.ndarray, resultado: dict) -> np.ndarray:
        """
        Desenha bounding boxes e informações na imagem.
        Útil para debug e demonstração no vídeo.
        """
        img_anotada = imagem.copy()

        # Cor baseada no resultado
        cor = (0, 200, 0) if resultado["presenca"] else (0, 0, 200)
        label = "PRESENCA DETECTADA" if resultado["presenca"] else "SEM PRESENCA"

        # Label no topo
        cv2.rectangle(img_anotada, (0, 0), (img_anotada.shape[1], 35), cor, -1)
        cv2.putText(img_anotada, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Bounding boxes (HOG)
        if "detalhe_hog" in resultado:
            for det in resultado["detalhe_hog"].get("deteccoes", []):
                x, y, w, h = det["bbox"]
                cv2.rectangle(img_anotada, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_anotada, f"HOG {det['confianca']:.2f}",
                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Bounding boxes (MediaPipe)
        if "detalhe_mediapipe" in resultado:
            for det in resultado["detalhe_mediapipe"].get("deteccoes", []):
                x, y, w, h = det["bbox"]
                cv2.rectangle(img_anotada, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img_anotada, f"Face {det['confianca']:.2f}",
                            (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Confiança geral
        conf_texto = f"Confianca: {resultado['confianca_max']:.0%}"
        cv2.putText(img_anotada, conf_texto, (10, img_anotada.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_anotada


def extrair_features(imagem: np.ndarray) -> np.ndarray:
    """
    Extrai features de uma imagem para o classificador de ML.

    Features extraídas:
    1. Histograma de cor (HSV) — 48 bins (16H + 16S + 16V)
    2. Estatísticas de textura — média e desvio de gradientes (2 valores)
    3. Proporção de pixels de pele — 1 valor
    4. HOG features reduzidas — 50 valores (PCA-like via resize)

    Total: 101 features por imagem
    """
    # Resize para normalizar
    img_resized = cv2.resize(imagem, (128, 96))

    # --- Feature 1: Histograma HSV ---
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    # Normalizar
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    # --- Feature 2: Estatísticas de gradiente (textura) ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    grad_stats = np.array([mag.mean(), mag.std()])

    # --- Feature 3: Proporção de pixels de pele ---
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.array([mask_skin.sum() / (255.0 * mask_skin.size)])

    # --- Feature 4: HOG compacto ---
    # Reduzir imagem a 64x48 e calcular HOG simplificado
    img_tiny = cv2.resize(gray, (64, 48))
    # Dividir em blocos 8x8 e calcular gradiente dominante
    hog_features = []
    for y in range(0, 48, 8):
        for x in range(0, 64, 8):
            bloco = img_tiny[y:y+8, x:x+8].astype(np.float32)
            gx = cv2.Sobel(bloco, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(bloco, cv2.CV_32F, 0, 1, ksize=3)
            mag_b = np.sqrt(gx**2 + gy**2)
            hog_features.append(mag_b.mean())
    hog_features = np.array(hog_features)
    # Normalizar
    hog_features = hog_features / (hog_features.max() + 1e-7)

    # Concatenar todas as features
    features = np.concatenate([hist_h, hist_s, hist_v, grad_stats, skin_ratio, hog_features])

    return features


def processar_dataset(dataset_dir: str = None) -> tuple:
    """
    Processa todas as imagens do dataset e extrai features.

    Returns:
        X: array de features (N, 101)
        y: array de labels (1=presença, 0=ausência)
        arquivos: lista de caminhos
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(BASE_DIR, "vision", "dataset")

    X_list = []
    y_list = []
    arquivos = []

    # Processar imagens com presença
    pasta_com = os.path.join(dataset_dir, "com_presenca")
    if os.path.exists(pasta_com):
        for nome in sorted(os.listdir(pasta_com)):
            if nome.endswith((".jpg", ".png", ".jpeg")):
                caminho = os.path.join(pasta_com, nome)
                img = cv2.imread(caminho)
                if img is not None:
                    features = extrair_features(img)
                    X_list.append(features)
                    y_list.append(1)
                    arquivos.append(caminho)

    # Processar imagens sem presença
    pasta_sem = os.path.join(dataset_dir, "sem_presenca")
    if os.path.exists(pasta_sem):
        for nome in sorted(os.listdir(pasta_sem)):
            if nome.endswith((".jpg", ".png", ".jpeg")):
                caminho = os.path.join(pasta_sem, nome)
                img = cv2.imread(caminho)
                if img is not None:
                    features = extrair_features(img)
                    X_list.append(features)
                    y_list.append(0)
                    arquivos.append(caminho)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, arquivos


# =============================================================================
# EXECUÇÃO STANDALONE (teste do detector)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DETECTOR DE PRESENÇA — TESTE")
    print("=" * 60)

    detector = DetectorPresenca(metodo="hog")  # HOG funciona melhor com silhuetas sintéticas

    dataset_dir = os.path.join(BASE_DIR, "vision", "dataset")
    resultados_dir = os.path.join(BASE_DIR, "vision", "resultados")
    os.makedirs(resultados_dir, exist_ok=True)

    acertos = 0
    total = 0

    for classe, label_real in [("com_presenca", True), ("sem_presenca", False)]:
        pasta = os.path.join(dataset_dir, classe)
        if not os.path.exists(pasta):
            continue

        print(f"\n  Testando {classe}/:")
        for nome in sorted(os.listdir(pasta)):
            if not nome.endswith((".jpg", ".png")):
                continue

            caminho = os.path.join(pasta, nome)
            resultado = detector.detectar_arquivo(caminho)

            correto = resultado["presenca"] == label_real
            acertos += int(correto)
            total += 1

            status = "✓" if correto else "✗"
            print(f"    {status} {nome:25s} → detectado={resultado['presenca']}, "
                  f"esperado={label_real}, conf={resultado['confianca_max']:.2f}")

            # Salvar imagem anotada
            img = cv2.imread(caminho)
            img_anotada = detector.anotar_imagem(img, resultado)
            cv2.imwrite(os.path.join(resultados_dir, f"anotado_{nome}"), img_anotada)

    print(f"\n  Acurácia do detector: {acertos}/{total} = {acertos/total:.1%}")
    print(f"  Imagens anotadas salvas em: {resultados_dir}/")

    # Extrair features do dataset
    print("\n" + "=" * 60)
    print("  EXTRAÇÃO DE FEATURES PARA ML")
    print("=" * 60)

    X, y, arquivos = processar_dataset()
    print(f"\n  Features extraídas: {X.shape}")
    print(f"  Com presença: {(y == 1).sum()}")
    print(f"  Sem presença: {(y == 0).sum()}")
    print(f"  Features por imagem: {X.shape[1]}")
