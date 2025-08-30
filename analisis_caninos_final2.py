# analisis_canino.py
import os
import pathlib
import pickle
import platform
import sys
import cv2
import torch # type: ignore 
import math
import numpy as np
from PIL import Image # type: ignore
from torchvision import transforms #type: ignore
from tooth_shape_model_unet import UNet
from pathlib import Path

plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# ============================
# VERIFICAR ARGUMENTO DE ENTRADA
# ============================
if len(sys.argv) != 2:
    print("Uso: python analisis_canino.py ruta/a/imagen.jpg")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"ERROR: No se encontró la imagen en {img_path}")
    sys.exit(1)

# ============================
# CONFIGURACIÓN GENERAL
# ============================
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES_IZQ = 2  # fondo y canino izquierdo
NUM_CLASSES_DER = 3  # fondo, canino izquierdo, canino derecho
CONF_THRESHOLD = 0.8
output_dir = "pruebas_modelo"
os.makedirs(output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# CARGA DE MODELOS
# ============================
model_izq = UNet(num_classes=NUM_CLASSES_IZQ).to(device)
model_izq.load_state_dict(torch.load("models/modelo4.pth", map_location=device))
model_izq.eval()

model_der = UNet(num_classes=NUM_CLASSES_DER).to(device)
model_der.load_state_dict(torch.load("models/modelo2.pth", map_location=device))
model_der.eval()


model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=Path('models/tooth_detection.pt')).to(device)
model_yolo.conf = CONF_THRESHOLD

infer_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# ============================
def calculate_line_intersection(line1_p1, line1_p2, line2_p1, line2_p2):
    """
    Calcula el punto de intersección entre dos líneas definidas por dos puntos cada una.
    Si las líneas son paralelas o no se intersectan dentro de los límites de la imagen, devuelve None.
    
    Args:
        line1_p1, line1_p2: Puntos que definen la primera línea
        line2_p1, line2_p2: Puntos que definen la segunda línea
        
    Returns:
        tuple: Coordenadas (x, y) del punto de intersección o None si no hay intersección
    """
    # Línea 1: (x1, y1) a (x2, y2)
    x1, y1 = line1_p1
    x2, y2 = line1_p2
    
    # Línea 2: (x3, y3) a (x4, y4)
    x3, y3 = line2_p1
    x4, y4 = line2_p2
    
    # Calcular denominador para verificar si las líneas son paralelas
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    
    if denom == 0:  # Líneas paralelas
        return None
    
    # Calcular el punto de intersección
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    
    # Punto de intersección
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (int(x), int(y))
def extend_line_to_boundaries(point1, point2, img_width, img_height, midline_x=None):
    """
    Extiende una línea definida por dos puntos hasta los límites de la imagen o hasta intersectar con la línea media.
    
    Args:
        point1 (tuple): Coordenadas (x, y) del primer punto.
        point2 (tuple): Coordenadas (x, y) del segundo punto.
        img_width (int): Ancho de la imagen.
        img_height (int): Alto de la imagen.
        midline_x (int, opcional): Coordenada x de la línea media. Si se proporciona, la línea se extenderá hasta esta línea.
        
    Returns:
        tuple: Un par de tuplas con las coordenadas de los puntos extendidos (p1_extended, p2_extended).
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Si los puntos son iguales, no se puede definir una dirección
    if x1 == x2 and y1 == y2:
        return point1, point2
    
    # Calcular la dirección de la línea
    dx = x2 - x1
    dy = y2 - y1
    
    # Si la línea es vertical
    if dx == 0:
        # Extender hasta los bordes superior e inferior
        return (x1, 0), (x1, img_height)
    
    # Calcular la pendiente y el intercepto
    m = dy / dx
    b = y1 - m * x1
    
    # Puntos extendidos
    extended_points = []
    
    # Si hay una línea media definida, calcular la intersección con ella
    if midline_x is not None:
        # Calcular el punto de intersección con la línea media
        y_intersect = m * midline_x + b
        
        # Verificar si la intersección está dentro de los límites de la imagen
        if 0 <= y_intersect <= img_height:
            # Determinar en qué lado de la línea media está el punto original
            if (x1 < midline_x and x2 < midline_x) or (x1 > midline_x and x2 > midline_x):
                # Ambos puntos están en el mismo lado de la línea media
                # Extender hasta la línea media en una dirección
                if x1 < midline_x:
                    extended_points.append((midline_x, int(y_intersect)))
                else:
                    extended_points.append((midline_x, int(y_intersect)))
            elif (x1 < midline_x and x2 > midline_x) or (x1 > midline_x and x2 < midline_x):
                # No es necesario extender hasta la línea media
                pass
    
    # Intersecciones con los bordes de la imagen
    
    # Intersección con y=0 (borde superior)
    if abs(m) > 0.0001:  # No es una línea horizontal
        x_top = (0 - b) / m
        if 0 <= x_top <= img_width:
            extended_points.append((int(x_top), 0))
    
    # Intersección con y=img_height (borde inferior)
    if abs(m) > 0.0001:  # No es una línea horizontal
        x_bottom = (img_height - b) / m
        if 0 <= x_bottom <= img_width:
            extended_points.append((int(x_bottom), img_height))
    
    # Intersección con x=0 (borde izquierdo)
    y_left = b
    if 0 <= y_left <= img_height:
        extended_points.append((0, int(y_left)))
    
    # Intersección con x=img_width (borde derecho)
    y_right = m * img_width + b
    if 0 <= y_right <= img_height:
        extended_points.append((img_width, int(y_right)))
    
    # Si no hay suficientes puntos de intersección, usar los puntos originales
    if len(extended_points) < 2:
        return point1, point2
    
    # Ordenar los puntos extendidos según su distancia desde el punto medio entre p1 y p2
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Si estamos del lado izquierdo de la línea media y queremos extender hacia la línea media
    if midline_x is not None and (x1 < midline_x and x2 < midline_x):
        # Encontrar el punto más cercano al borde y el punto más cercano a la línea media
        points_sorted = sorted(extended_points, key=lambda p: p[0])  # Ordenar por coordenada x
        return points_sorted[0], points_sorted[-1]  # El primero es el más a la izquierda, el último es el más a la derecha
    
    # Si estamos del lado derecho de la línea media y queremos extender hacia la línea media
    elif midline_x is not None and (x1 > midline_x and x2 > midline_x):
        # Encontrar el punto más cercano al borde y el punto más cercano a la línea media
        points_sorted = sorted(extended_points, key=lambda p: p[0], reverse=True)  # Ordenar por coordenada x (reverso)
        return points_sorted[0], points_sorted[-1]  # El primero es el más a la derecha, el último es el más a la izquierda
    
    # En otros casos, simplemente usar las dos intersecciones más alejadas entre sí
    else:
        # Calcular todas las combinaciones de distancias entre puntos
        max_dist = 0
        p1_ext, p2_ext = extended_points[0], extended_points[1]
        
        for i in range(len(extended_points)):
            for j in range(i + 1, len(extended_points)):
                dist = math.sqrt((extended_points[i][0] - extended_points[j][0])**2 + 
                                (extended_points[i][1] - extended_points[j][1])**2)
                if dist > max_dist:
                    max_dist = dist
                    p1_ext, p2_ext = extended_points[i], extended_points[j]
        
        return p1_ext, p2_ext
def get_center(detection):
    xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], 
                                      detection['xmax'], detection['ymax']])
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    return (cx, cy)
def get_corners(detection, side):
    if side == 'izq':
        return (int(detection['xmax']), int(detection['ymax'])), (int(detection['xmin']), int(detection['ymin']))
    elif side == 'der':
        return (int(detection['xmin']), int(detection['ymax'])), (int(detection['xmax']), int(detection['ymin']))
    else:   
        raise ValueError("Lado no válido. Debe ser 'izq' o 'der'.")
def segmentar_canino_por_lado(roi_bgr, lado, model_izq, model_der):
    """
    Segmenta el ROI del canino usando el modelo correspondiente (izquierdo o derecho).
    Devuelve una máscara binaria. Si el modelo no tiene suficientes canales, retorna una máscara vacía.
    """
    modelo = model_izq if lado == 'izq' else model_der

    img_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = infer_transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = modelo(x)
        probs = torch.softmax(logits, dim=1)[0]

        try:
            canal = probs[1] if lado == 'izq' else probs[2]
        except IndexError:
            print(f"[ERROR] Modelo no tiene canal para el lado '{lado}'. Usando máscara vacía.")
            return np.zeros((roi_bgr.shape[0], roi_bgr.shape[1]), dtype=np.uint8)

        mask = (canal.cpu().numpy() > 0.03).astype(np.uint8) * 255

    h, w = roi_bgr.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

def calculate_angle(line_p1, line_p2, vertical_line_x):
    """
    Calcula el ángulo entre una línea definida por dos puntos y una línea vertical.
    
    Args:
        line_p1 (tuple): Primer punto de la línea.
        line_p2 (tuple): Segundo punto de la línea.
        vertical_line_x (int): Coordenada x de la línea vertical.
        
    Returns:
        float: Ángulo en grados entre las líneas.
    """
    # Verificar que los puntos no sean iguales
    if line_p1[0] == line_p2[0] and line_p1[1] == line_p2[1]:
        return 0  # No se puede calcular el ángulo si los puntos son iguales
    
    # Vector de la línea
    vector_line = (line_p2[0] - line_p1[0], line_p2[1] - line_p1[1])
    
    # Vector de la línea vertical (0, 1) normalizado
    vector_vertical = (0, 1)
    
    # Calcular el ángulo entre los vectores usando el producto punto
    # Normalizar los vectores
    magnitude_line = math.sqrt(vector_line[0]**2 + vector_line[1]**2)
    
    if magnitude_line == 0:
        return 0
    
    unit_vector_line = (vector_line[0] / magnitude_line, vector_line[1] / magnitude_line)
    
    # Producto punto de los vectores unitarios
    dot_product = unit_vector_line[0] * vector_vertical[0] + unit_vector_line[1] * vector_vertical[1]
    
    # Asegurarse de que el producto punto esté en el rango [-1, 1]
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calcular el ángulo en radianes y convertirlo a grados
    angle_rad = math.acos(dot_product)
    angle_deg = math.degrees(angle_rad)
    
    # Determinar la dirección del ángulo (positivo o negativo)
    # Si el punto p2 está a la derecha de la línea vertical, el ángulo es positivo
    # Si está a la izquierda, el ángulo es negativo
    direction = 1 if (line_p1[0] < vertical_line_x and line_p2[0] > vertical_line_x) or \
                    (line_p1[0] > vertical_line_x and line_p2[0] < vertical_line_x and line_p1[1] > line_p2[1]) else -1
    
    # Ajustar el ángulo según el cuadrante
    if unit_vector_line[0] < 0:
        angle_deg = 180 - angle_deg
    
    # Asegurarse de que el ángulo esté entre 0 y 180 grados
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
        
    return angle_deg * direction
def process_and_draw_canine(det, orig_img, side, inc_center_x, model_izq, model_der):
    """
    Procesa un canino detectado: segmenta con el modelo correspondiente,
    calcula el eje principal y el ángulo con la línea media, y dibuja todo.
    
    Args:
        det: Fila del DataFrame con la detección del canino.
        orig_img: Imagen BGR original.
        side: 'izq' o 'der'.
        inc_center_x: Coordenada x de la línea media vertical.
        model_izq: Modelo U-Net para canino izquierdo.
        model_der: Modelo U-Net para canino derecho.
    
    Returns:
        float: Ángulo del canino respecto a la línea media.
    """
    coords = (int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax']))
    x1, y1, x2, y2 = coords
    roi = orig_img[y1:y2, x1:x2]

    # 1) Segmentar usando el modelo correcto
    mask_roi = segmentar_canino_por_lado(roi, side, model_izq, model_der)
    cv2.imwrite(f"debug/mask_roi_{side}.png", mask_roi)

    # 2) Extraer contorno
    cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print(f"No se encontró contorno para canino {side}")
        return None

    contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)

    # 3) PCA sobre el contorno
    mean, vecs, _ = cv2.PCACompute2(contour, mean=None)
    axis = vecs[0]
    dif = contour - mean
    projs = dif.dot(axis.T)

    p1 = list(contour[np.argmin(projs)].astype(int))
    p2 = list(contour[np.argmax(projs)].astype(int))
    if p1[1] > p2[1]: p1, p2 = p2, p1

    # 4) Ajuste de orientación
    if (side == "izq") and (p1[0] > p2[0]):
        p1[1], p2[1] = p2[1], p1[1]
    if (side == "der") and (p1[0] < p2[0]):
        p1[1], p2[1] = p2[1], p1[1]

    # 5) Convertir a coordenadas globales
    p1g = (p1[0] + x1, p1[1] + y1)
    p2g = (p2[0] + x1, p2[1] + y1)

    # 6) Dibujar extremos
    cv2.circle(orig_img, p1g, 4, (0, 255, 0), -1)
    cv2.circle(orig_img, p2g, 4, (0, 255, 0), -1)
    cv2.putText(orig_img, 'p1', (p1g[0] + 10, p1g[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(orig_img, 'p2', (p2g[0] + 10, p2g[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 7) Extender línea hasta bordes
    h, w = orig_img.shape[:2]
    exp1, exp2 = extend_line_to_boundaries(p1g, p2g, w, h, inc_center_x)
    cv2.line(orig_img, exp1, exp2, (0, 0, 255), 2)

    # 8) Calcular intersección y ángulo
    midline_top = (inc_center_x, 0)
    midline_bottom = (inc_center_x, h)
    intersection = calculate_line_intersection(exp1, exp2, midline_top, midline_bottom)

    if intersection:
        cv2.circle(orig_img, intersection, 6, (255, 0, 255), -1)
        angle = calculate_angle(exp1, exp2, inc_center_x)
        angle_text = f"Angulo: {abs(round(angle, 2))} deg"
        text_x = (x1 + x2) // 2 - 50
        text_y = y2 + 30

        cv2.putText(orig_img, angle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Dibujar arco del ángulo
        radius = 40
        start_angle = 90
        end_angle = 90 + angle if angle > 0 else 90 - abs(angle)
        cv2.ellipse(orig_img, intersection, (radius, radius), 0,
                    min(start_angle, end_angle), max(start_angle, end_angle),
                    (255, 0, 255), 2)

        return angle

    return None
# ============================
# PROCESAMIENTO
# ============================
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No se pudo leer la imagen en: {img_path}")

orig = img.copy()
height, width = img.shape[:2]

df = model_yolo(img).pandas().xyxy[0]
detections_inc = df[df['name'] == 'inc']
detections_can = df[df['name'] == 'canine']

# Línea media
if len(detections_inc) == 0:
    print("No se detectó incisivo. Se aborta.")
    sys.exit(1)

inc = detections_inc.sort_values('confidence', ascending=False).iloc[0]
inc_center = get_center(inc)
cv2.line(orig, (inc_center[0], 0), (inc_center[0], height), (0, 255, 0), 2)
cv2.putText(orig, 'Línea media', (inc_center[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Procesar caninos
angulo_derecho = None
angulo_izquierdo = None
riesgo_derecho = False
riesgo_izquierdo = False

for _, det in detections_can.iterrows():
    center = get_center(det)
    lado = 'izq' if center[0] < inc_center[0] else 'der'
    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
    color = (255, 0, 0) if lado == 'izq' else (0, 0, 255)
    
    cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
    cv2.circle(orig, center, 5, color, -1)
    
    try:
        angle = process_and_draw_canine(det, orig, lado, inc_center[0], model_izq, model_der)
        if isinstance(angle, (int, float, np.float32, np.float64)):
            if lado == 'der':
                angulo_derecho = float(angle)
                riesgo_derecho = abs(angulo_derecho) > 15
            else:
                angulo_izquierdo = float(angle)
                riesgo_izquierdo = abs(angulo_izquierdo) > 15
    except Exception as e:
        print(f"Error al procesar canino {lado}: {e}")
        continue

# Construir nombre del archivo con información completa
base_name = os.path.splitext(os.path.basename(img_path))[0]
nombre_partes = [base_name]

# Agregar información de ángulos
if angulo_izquierdo is not None:
    riesgo_str = "R" if riesgo_izquierdo else "N"
    nombre_partes.append(f"izq{abs(angulo_izquierdo):.1f}{riesgo_str}")
else:
    nombre_partes.append(f"izq0.0N")

if angulo_derecho is not None:
    riesgo_str = "R" if riesgo_derecho else "N"
    nombre_partes.append(f"der{abs(angulo_derecho):.1f}{riesgo_str}")
else:
    nombre_partes.append(f"der0.0N")

# Determinar riesgo general
riesgo_general = riesgo_izquierdo or riesgo_derecho

# Agregar estado general de riesgo
estado_general = "RIESGO" if riesgo_general else "NORMAL"
nombre_partes.append(estado_general)

# Construir nombre final
nombre_salida = "_".join(nombre_partes) + ".jpg"

# Guardar salida
os.makedirs(output_dir, exist_ok=True)
ruta_completa = os.path.join(output_dir, nombre_salida)
cv2.imwrite(ruta_completa, orig)
print(f"Resultado guardado en: {output_dir}/{nombre_salida}")

# Mostrar resultados
if angulo_izquierdo is not None:
    print(f"Ángulo canino izquierdo: {abs(round(angulo_izquierdo, 2))}°")
    if riesgo_izquierdo:
        print("Canino izquierdo: RIESGO")
else:
    print("No se pudo calcular el ángulo del canino izquierdo.")

if angulo_derecho is not None:
    print(f"Ángulo canino derecho: {abs(round(angulo_derecho, 2))}°")
    if riesgo_derecho:
        print("Canino derecho: RIESGO")
else:
    print("No se pudo calcular el ángulo del canino derecho.")

if riesgo_general:
    print("EXISTE RIESGO")
else:
    print("NO EXISTE RIESGO")