import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import random

# Dataset artificial de entrenamiento (supervisado)
data = {
    "origen": ["Portal Norte", "Héroes", "Calle 45", "Calle 26", "Calle 13", "Bosa", "Portal Sur", "Calle 45", "Calle 13", "Héroes"],
    "destino": ["Héroes", "Calle 45", "Calle 26", "Calle 13", "Bosa", "Portal Sur", "Calle 26", "Bosa", "Portal Sur", "Portal Norte"],
    "distancia_km": [4.5, 3.2, 2.5, 1.8, 5.6, 4.0, 6.3, 8.7, 9.6, 7.7],
    "duracion_min": [10, 8, 6, 5, 15, 12, 17, 22, 25, 20],
    "congestion": ["media", "baja", "media", "alta", "alta", "media", "alta", "alta", "media", "baja"],
    "transbordos": [0, 1, 0, 0, 2, 1, 2, 3, 2, 1],
    "ruta_optima": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# Codificar la congestión
label_encoder = LabelEncoder()
df["congestion_cod"] = label_encoder.fit_transform(df["congestion"])

# Seleccionar features y etiqueta
X = df[["distancia_km", "duracion_min", "congestion_cod", "transbordos"]]
y = df["ruta_optima"]

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
print(f"Precisión del modelo: {accuracy_score(y_test, modelo.predict(X_test)):.2f}")

# Guardar el modelo
joblib.dump(modelo, "modelo_rutas.pkl")

# Mapa base de distancias estimadas entre estaciones (para simular)
distancia_base = {
    ("Portal Norte", "Héroes"): 4.5,
    ("Héroes", "Calle 45"): 3.2,
    ("Calle 45", "Calle 26"): 2.5,
    ("Calle 26", "Calle 13"): 1.8,
    ("Calle 13", "Bosa"): 5.6,
    ("Bosa", "Portal Sur"): 4.0,
    ("Portal Sur", "Calle 26"): 6.3,
    ("Calle 45", "Bosa"): 8.7,
    ("Calle 13", "Portal Sur"): 9.6,
    ("Héroes", "Portal Norte"): 7.7,
}

# Simulador de valores basado en origen/destino
def simular_ruta(origen, destino):
    key = (origen, destino)
    key_inv = (destino, origen)
    
    if key in distancia_base:
        distancia = distancia_base[key]
    elif key_inv in distancia_base:
        distancia = distancia_base[key_inv]
    else:
        distancia = round(random.uniform(3.0, 10.0), 1)

    # Simulación sencilla
    duracion = round(distancia * random.uniform(1.5, 3.0))  # Ej: 4 km -> 8 a 12 min
    transbordos = random.randint(0, 3)
    
    if duracion > 20 or transbordos > 2:
        congestion = "alta"
    elif duracion > 10:
        congestion = "media"
    else:
        congestion = "baja"

    congestion_cod = label_encoder.transform([congestion])[0]
    
    return distancia, duracion, congestion_cod, transbordos, congestion

# Función principal de predicción
def predecir_ruta_con_origen_destino():
    print("=== PREDICCIÓN DE RUTA ÓPTIMA ===")
    origen = input("Ingrese estación de origen: ").strip()
    destino = input("Ingrese estación de destino: ").strip()

    distancia, duracion, congestion_cod, transbordos, congest = simular_ruta(origen, destino)
    
    print("\n Simulación de valores:")
    print(f"Distancia estimada: {distancia} km")
    print(f"Duración estimada: {duracion} minutos")
    print(f"Nivel de congestión: {congest}")
    print(f"Número de transbordos: {transbordos}\n")
    
    entrada = [[distancia, duracion, congestion_cod, transbordos]]
    resultado = modelo.predict(entrada)

    if resultado[0] == 1:
        print("Ruta óptima.")
    else:
        print("Ruta NO óptima.")

# Descomenta para ejecutar desde consola
predecir_ruta_con_origen_destino()
