import base64
import glob
from pathlib import Path
import re
import shutil
import subprocess
import bcrypt
import cv2
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import os
from typing import Optional
from PIL import Image
import io
import io
import torch
import torch.nn as nn
from torchvision import models, transforms

from dotenv import load_dotenv
load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY', 'your_secret_key_here')
if not SECRET_KEY:
    raise ValueError("No se ha definido la variable de entorno SECRET_KEY")
ALGORITHM = os.getenv('ALGORITHM', 'HS256')
if not ALGORITHM:
    raise ValueError("No se ha definido la variable de entorno ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))
API_PORT = int(os.getenv('API_PORT', 33110))

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'database': os.getenv('DB_NAME', 'radiografias_db'),
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}

# Inicializar FastAPI
app = FastAPI(title="API de Radiografías", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de seguridad
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Cargar modelo entrenado
def load_model(model_path: str, num_classes: int = 2) -> nn.Module:
    model = models.resnet18(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    # Ajustar capa final
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    # Cargar pesos
    state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

# modelo para filtrar/verificar que la imagen que se sube es una radiografía
FILTER_MODEL_PATH = os.getenv('FILTER_MODEL_PATH', 'models/filter/rx_classifier_final.pth')
filter_model = load_model(FILTER_MODEL_PATH)

# Etiquetas correspondencia index->clase
CLASS_MAP = {0: 'non_panoramic', 1: 'panoramic'}

# Modelos Pydantic
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    token: Optional[str] = None

class XRayAnalysis(BaseModel):
    patient_age: int
    patient_gender: str
    doctor_observations: str
    analysis_result: str
    timestamp: datetime

# Funciones de utilidad para base de datos MySQL
def get_db_connection():
    """Crear conexión a MySQL"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error conectando a MySQL: {e}")
        raise HTTPException(status_code=500, detail="Error de conexión a la base de datos")

def init_db():
    """Inicializar base de datos MySQL"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Crear base de datos si no existe
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Tabla de usuarios
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_username (username),
                INDEX idx_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Tabla de análisis de radiografías
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS xray_analyses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                patient_age INT NOT NULL,
                patient_gender ENUM('M', 'F') NOT NULL,
                doctor_observations TEXT,
                image_path VARCHAR(500) NOT NULL,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                INDEX idx_user_id (user_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        # Tabla de retroalimentación
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INT AUTO_INCREMENT PRIMARY KEY,
                analysis_id INT NOT NULL,
                user_id INT NOT NULL,
                feedback_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES xray_analyses (id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                INDEX idx_analysis_id (analysis_id),
                INDEX idx_user_id (user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        ''')
        
        connection.commit()
        
        admin_user = {
            "username": os.getenv("MAIN_USER_USERNAME"),
            "email":    os.getenv("MAIN_USER_EMAIL"),
            "full_name": os.getenv("MAIN_USER_NAME"),
            "password": os.getenv("MAIN_USER_PASSWORD"),
        }
        # 2a) ¿Existe ya?
        cursor.execute(
            "SELECT COUNT(*) FROM users WHERE username = %s OR email = %s",
            (admin_user["username"], admin_user["email"])
        )
        (count,) = cursor.fetchone()
        
        if count == 0:
            # 2b) Si no existe, lo insertamos
            password_hash = bcrypt.hashpw(admin_user["password"].encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                """
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (%s, %s, %s, %s)
                """,
                (admin_user["username"],
                 admin_user["email"],
                 password_hash,
                 admin_user["full_name"])
            )
            connection.commit()
            print("Usuario admin creado correctamente.")
        else:
            print("Usuario admin ya existe, no se crea de nuevo.")
        print("Base de datos inicializada correctamente")
        
    except Error as e:
        print(f"Error inicializando base de datos: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail="Error inicializando base de datos")
    finally:
        cursor.close()
        connection.close()

# Funciones de autenticación
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user_by_username(username: str):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT id, username, email, password_hash, full_name, created_at FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        return user
    except Error as e:
        print(f"Error obteniendo usuario por username: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_user_by_email(email: str):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        cursor.execute("SELECT id, username, email, password_hash, full_name, created_at FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        return user
    except Error as e:
        print(f"Error obteniendo usuario por email: {e}")
        return None
    finally:
        cursor.close()
        connection.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        print(username)
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")
    
    user = get_user_by_email(username)
    if user is None:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user

# Función para verificar si una imagen es una radiografía
def is_xray_image(image_bytes: bytes) -> bool:
    """
    Verifica si una imagen es una radiografía basándose en características visuales.
    Esta es una implementación básica que puede mejorarse con ML.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0) 

        with torch.no_grad():
            outputs = filter_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        label = CLASS_MAP[int(pred_idx)]
        
        if label == 'panoramic' and conf > 0.85:
            return True
        return False
    except Exception as e:
        print(f"Error verificando imagen: {e}")
        return False
class AnalysisResult:
        def __init__(self, success: bool, processed_image: Optional[np.ndarray] = None, angles: Optional[list] = None):
            self.success = success
            self.has_risk = False
            self.processed_image = processed_image
            self.angles = angles if angles is not None else []


def analyze_xray_via_cli(path, patient_age: int, patient_gender: str, doctor_observations: str) -> AnalysisResult:
    input_path = Path(path)
    base_name = input_path.stem 

    proc = subprocess.run(
        ["python3", "analisis_caninos_final2.py", str(input_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    log = proc.stdout or proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Error: {log}")

    # 2) Buscar el archivo de salida con glob
    resultados_dir = Path("pruebas_modelo")
    pattern = str(resultados_dir / f"{base_name}_*.jpg")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No se encontró ningún resultado para {base_name} en {resultados_dir}")
    # Si hay varios, nos quedamos con el último (más reciente)
    output_path = Path(sorted(matches)[-1])

    # 3) Parsear la info del nombre:
    #    nombre = base_name_izq12.3R_der05.6N_NORMAL.jpg
    regex = re.compile(
        rf"^{re.escape(base_name)}_"
        r"izq(?P<ang_izq>\d+\.\d+)(?P<r_izq>[RN])_"
        r"der(?P<ang_der>\d+\.\d+)(?P<r_der>[RN])_"
        r"(?P<estado>RIESGO|NORMAL)\.jpg$"
    )
    m = regex.match(output_path.name)
    if not m:
        raise ValueError(f"Nombre de archivo inesperado: {output_path.name}")

    ang_izq = float(m.group("ang_izq"))
    riesgo_izq = (m.group("r_izq") == "R")
    ang_der = float(m.group("ang_der"))
    riesgo_der = (m.group("r_der") == "R")
    estado_general = (m.group("estado") == "RIESGO")

    # 4) Preparamos la lista de ángulos con su riesgo
    angles_data = [
        {"side": "izq", "angle": ang_izq, "at_risk": riesgo_izq},
        {"side": "der", "angle": ang_der, "at_risk": riesgo_der},
    ]

    # 5) Leemos la imagen resultante
    image_bytes_out = output_path.read_bytes()

    # 6) Devolvemos todos los datos en el AnalysisResult
    return AnalysisResult(
        success=True,
        processed_image=image_bytes_out,
        angles=angles_data,
    )

@app.on_event("startup")
async def startup_event():
    init_db()
    os.makedirs("uploads", exist_ok=True)

@app.post("/api/auth/register", response_model=dict)
async def register_user(user: UserRegister, _: int = Depends(get_current_user)):
    """Registrar un nuevo usuario"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar si el usuario o email ya existe
        cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", 
                      (user.username, user.email))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Usuario o email ya existe")
        
        password_hash = get_password_hash(user.password)
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (%s, %s, %s, %s)
        """, (user.username, user.email, password_hash, user.full_name))
        
        connection.commit()
        return {"message": "Usuario registrado exitosamente"}
        
    except mysql.connector.IntegrityError:
        raise HTTPException(status_code=400, detail="Usuario o email ya existe")
    except Error as e:
        print(f"Error registrando usuario: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    finally:
        cursor.close()
        connection.close()

@app.post("/api/auth/login", response_model=User)
async def login(user: UserLogin):
    """Iniciar sesión y obtener token JWT"""
    db_user = get_user_by_email(user.username)
    
    if not db_user or not verify_password(user.password, db_user[3]):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "id": db_user[0],
        "username": db_user[1],
        "email": db_user[2],
        "full_name": db_user[4],
        "token": access_token
    }

@app.post("/api/analyze")
async def analyze_xray_endpoint(
    file: UploadFile = File(...),
    patient_age: int = Form(...),
    patient_gender: str = Form(...),
    doctor_observations: str = Form(""),
    current_user = Depends(get_current_user)
):
    """Analizar radiografía con verificación de imagen"""

    # 1) Validaciones básicas
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "El archivo debe ser una imagen")
    if patient_age < 0 or patient_age > 150:
        raise HTTPException(400, "Edad del paciente inválida")
    if patient_gender.upper() not in ["M", "F"]:
        raise HTTPException(400, "Género debe ser: M o F")

    # 2) Leer bytes y guardarlos directamente
    image_bytes = await file.read()
    # if not is_xray_image(image_bytes):
    #     raise HTTPException(400, "La imagen no parece ser una radiografía válida")

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{current_user[0]}_{datetime.now():%Y%m%d_%H%M%S}.{ext}"
    file_path = uploads_dir / filename
    file_path.write_bytes(image_bytes)

    try: 
        analysis = analyze_xray_via_cli(str(file_path), patient_age, patient_gender.upper(), doctor_observations)
        if not analysis.success:
            raise HTTPException(500, "Error en el análisis de la radiografía")
    except RuntimeError as e:
        raise HTTPException(500, f"{e}")

    # 4) Guardar metadata en BD (sin volver a codificar la imagen)
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO xray_analyses
            (user_id, patient_age, patient_gender, doctor_observations, image_path)
            VALUES (%s, %s, %s, %s, %s)
        """, (current_user[0], patient_age, patient_gender.upper(), doctor_observations, str(file_path)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error guardando en BD: {e}")
    finally:
        cur.close()
        conn.close()

    # 5) Preparar la imagen resultante como Base64
    #    Aquí asumimos que analysis.processed_image ya es un `bytes` JPEG
    img_b64 = base64.b64encode(analysis.processed_image).decode('utf-8')
    data_uri = f"data:image/jpeg;base64,{img_b64}"

    # 6) Responder
    return JSONResponse({
        "image": data_uri,
        "summary": analysis.angles,
    })

@app.post("/api/feedback")
async def submit_feedback(
    analysis_id: int = Form(...),
    feedback: str = Form(...),
    current_user = Depends(get_current_user)
):
    """Enviar retroalimentación sobre un análisis de radiografía"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar que el análisis exista y pertenezca al usuario
        cursor.execute("SELECT id FROM xray_analyses WHERE id = %s AND user_id = %s", 
                       (analysis_id, current_user[0]))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Análisis no encontrado o no pertenece al usuario")
        
        # Guardar retroalimentación
        cursor.execute("""
            INSERT INTO feedback (analysis_id, user_id, feedback_text)
            VALUES (%s, %s, %s)
        """, (analysis_id, current_user[0], feedback))
        
        connection.commit()
        return {"message": "Retroalimentación enviada exitosamente"}
        
    except Error as e:
        print(f"Error enviando feedback: {e}")
        connection.rollback()
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    finally:
        cursor.close()
        connection.close()

@app.get("/api/analyses")
async def get_user_analyses(current_user = Depends(get_current_user)):
    """Obtener historial de análisis del usuario"""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
            SELECT id, patient_age, patient_gender, doctor_observations, 
                   analysis_result, created_at
            FROM xray_analyses 
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (current_user[0],))
        
        analyses = cursor.fetchall()
        
        return {
            "analyses": [
                {
                    "id": analysis[0],
                    "patient_age": analysis[1],
                    "patient_gender": analysis[2],
                    "doctor_observations": analysis[3],
                    "analysis_result": analysis[4],
                    "created_at": analysis[5].isoformat() if analysis[5] else None
                } for analysis in analyses
            ]
        }
        
    except Error as e:
        print(f"Error obteniendo análisis: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)