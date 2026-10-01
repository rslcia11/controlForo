import streamlit as st
import cv2
from ultralytics import YOLO
from collections import deque
import statistics
import numpy as np
from PIL import Image
import time
import math

# Configuración de la página
st.set_page_config(
    page_title="Control de Aforo Dinámico",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Sistema de Control de Aforo Dinámico")
st.markdown("---")

# Función para cargar el modelo (con cache para mejor rendimiento)
@st.cache_resource
def load_model():
    """Carga el modelo YOLO con cache para evitar recargas innecesarias"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Inicialización del estado de la sesión
if 'monitor_history' not in st.session_state:
    st.session_state.monitor_history = deque(maxlen=30)
if 'occupied_history' not in st.session_state:
    st.session_state.occupied_history = deque(maxlen=30)
if 'available_history' not in st.session_state:
    st.session_state.available_history = deque(maxlen=30)
if 'detection_counters' not in st.session_state:
    st.session_state.detection_counters = {
        'total_persons': 0,
        'total_monitors': 0,
        'total_laptops': 0,
        'occupied_stations': 0,
        'available_stations': 0
    }
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'camera_initialized' not in st.session_state:
    st.session_state.camera_initialized = False

# Sidebar con controles
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Control de la cámara
    camera_source = st.selectbox(
        "Fuente de video:",
        options=[0, 1, 2],
        format_func=lambda x: f"Cámara {x}",
        help="Selecciona la cámara a utilizar"
    )
    
    # Configuración de detección
    st.subheader("🎯 Configuración de Detección")
    
    proximity_threshold = st.slider(
        "Umbral de proximidad (pixels):",
        min_value=50,
        max_value=200,
        value=100,
        help="Distancia máxima para considerar que una persona está usando un equipo"
    )
    
    confidence_threshold = st.slider(
        "Umbral de confianza:",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        help="Confianza mínima para considerar una detección válida"
    )
    
    # Botón para resetear contadores
    if st.button("🔄 Resetear Contadores"):
        st.session_state.detection_counters = {
            'total_persons': 0,
            'total_monitors': 0,
            'total_laptops': 0,
            'occupied_stations': 0,
            'available_stations': 0
        }
    # Configuración del historial
    history_length = st.slider(
        "Tamaño del historial (frames):",
        min_value=10,
        max_value=60,
        value=30,
        help="Número de frames para estabilizar el conteo"
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Iniciar", type="primary")
    with col2:
        stop_button = st.button("⏹️ Detener", type="secondary")
    
    st.markdown("---")
    
    # Información del sistema
    st.subheader("📊 Estado del Sistema")
    status_placeholder = st.empty()
    
    # Métricas en tiempo real
    st.subheader("📈 Métricas en Tiempo Real")
    metrics_placeholder = st.empty()
    
    # Contadores acumulativos
    st.subheader("🔢 Contadores Totales")
    counters_placeholder = st.empty()

# Área principal de contenido
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video en Tiempo Real")
    video_placeholder = st.empty()

with col2:
    st.subheader("📋 Panel de Control")
    
    # Contenedores para métricas
    aforo_container = st.container()
    personas_container = st.container()
    equipos_container = st.container()
    ocupacion_container = st.container()
    
    # Gráfico histórico (placeholder para futuras mejoras)
    st.subheader("📊 Estado de Ocupación")
    chart_placeholder = st.empty()

def calculate_distance(box1, box2):
    """Calcula la distancia entre dos cajas de detección"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return math.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)

def process_frame(frame, model, proximity_threshold=100, confidence_threshold=0.5):
    """Procesa un frame individual y devuelve las detecciones con análisis de ocupación"""
    if model is None:
        return frame, 0, 0, 0, 0, 0, 0
    
    # Realizamos la detección de objetos
    results = model(frame, stream=True, verbose=False)
    
    # Listas para almacenar detecciones
    persons = []
    monitors = []
    laptops = []
    
    # Contadores
    person_count = 0
    monitor_count = 0
    laptop_count = 0
    
    # Analizamos los resultados de la detección
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            # Solo consideramos detecciones con confianza alta
            if confidence < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = (x1, y1, x2, y2)
            
            # Clasificamos y guardamos detecciones
            if class_name == 'person':
                person_count += 1
                persons.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                })
                # Dibujamos un recuadro verde para las personas
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Persona ({confidence:.2f})", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            elif class_name == 'tv':  # YOLO identifica monitores como 'tv'
                monitor_count += 1
                monitors.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'occupied': False
                })
                
            elif class_name == 'laptop':  # Laptops
                laptop_count += 1
                laptops.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                    'occupied': False
                })
    
    # Análisis de ocupación: verificar qué equipos están siendo usados
    occupied_stations = 0
    available_stations = 0
    
    # Verificar monitores
    for monitor in monitors:
        for person in persons:
            distance = calculate_distance(monitor['bbox'], person['bbox'])
            if distance <= proximity_threshold:
                monitor['occupied'] = True
                break
        
        # Dibujar monitor con color según ocupación
        x1, y1, x2, y2 = monitor['bbox']
        if monitor['occupied']:
            occupied_stations += 1
            color = (0, 165, 255)  # Naranja para ocupado
            status = "OCUPADO"
        else:
            available_stations += 1
            color = (255, 0, 0)  # Azul para disponible
            status = "LIBRE"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Monitor {status} ({monitor['confidence']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Verificar laptops
    for laptop in laptops:
        for person in persons:
            distance = calculate_distance(laptop['bbox'], person['bbox'])
            if distance <= proximity_threshold:
                laptop['occupied'] = True
                break
        
        # Dibujar laptop con color según ocupación
        x1, y1, x2, y2 = laptop['bbox']
        if laptop['occupied']:
            occupied_stations += 1
            color = (0, 165, 255)  # Naranja para ocupado
            status = "OCUPADO"
        else:
            available_stations += 1
            color = (255, 0, 0)  # Azul para disponible
            status = "LIBRE"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Laptop {status} ({laptop['confidence']:.2f})", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Actualizar contadores acumulativos
    st.session_state.detection_counters['total_persons'] += person_count
    st.session_state.detection_counters['total_monitors'] += monitor_count
    st.session_state.detection_counters['total_laptops'] += laptop_count
    st.session_state.detection_counters['occupied_stations'] += occupied_stations
    st.session_state.detection_counters['available_stations'] += available_stations
    
    total_equipment = monitor_count + laptop_count
    
    return frame, person_count, total_equipment, occupied_stations, available_stations, monitor_count, laptop_count

def update_display(person_count, total_equipment, occupied_stations, available_stations, 
                  monitor_count, laptop_count):
    """Actualiza la información mostrada en la interfaz"""
    
    # Actualizar métricas principales
    with aforo_container:
        if available_stations == 0 and total_equipment > 0:
            st.error(f"🔴 **TODOS LOS PUESTOS OCUPADOS** - 0/{total_equipment} puestos libres")
        elif available_stations <= 2 and total_equipment > 0:
            st.warning(f"🟡 **POCOS PUESTOS LIBRES** - {available_stations}/{total_equipment} puestos disponibles")
        else:
            st.success(f"🟢 **PUESTOS DISPONIBLES** - {available_stations}/{total_equipment} puestos libres")
    
    with personas_container:
        st.metric("👥 Personas Detectadas", person_count)
    
    with equipos_container:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🖥️ Monitores", monitor_count)
        with col2:
            st.metric("💻 Laptops", laptop_count)
    
    with ocupacion_container:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Puestos Ocupados", occupied_stations)
        with col2:
            st.metric("🟢 Puestos Libres", available_stations)
    
    # Actualizar métricas en sidebar
    with metrics_placeholder:
        st.metric("Total Equipos", total_equipment)
        if total_equipment > 0:
            ocupacion_pct = (occupied_stations / total_equipment) * 100
            st.metric("% Ocupación", f"{ocupacion_pct:.1f}%")
        else:
            st.metric("% Ocupación", "0.0%")
        
        # Eficiencia (personas por equipo ocupado)
        if occupied_stations > 0:
            eficiencia = person_count / occupied_stations
            st.metric("Personas/Puesto Ocupado", f"{eficiencia:.1f}")
    
    # Actualizar contadores totales
    with counters_placeholder:
        counters = st.session_state.detection_counters
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👥 Total Personas", counters['total_persons'])
            st.metric("🖥️ Total Monitores", counters['total_monitors'])
        with col2:
            st.metric("💻 Total Laptops", counters['total_laptops'])
            total_detections = counters['occupied_stations'] + counters['available_stations']
            st.metric("🔢 Total Detecciones", total_detections)

# Lógica principal
if start_button:
    st.session_state.is_running = True
    st.session_state.monitor_history = deque(maxlen=history_length)
    st.session_state.occupied_history = deque(maxlen=history_length)
    st.session_state.available_history = deque(maxlen=history_length)

if stop_button:
    st.session_state.is_running = False
    st.session_state.camera_initialized = False

# Actualizar estado en sidebar
with status_placeholder:
    if st.session_state.is_running:
        st.success("🟢 Sistema Activo")
    else:
        st.info("🔵 Sistema Detenido")

# Bucle principal de procesamiento
if st.session_state.is_running:
    # Cargar modelo
    model = load_model()
    if model is None:
        st.error("❌ No se pudo cargar el modelo YOLO. Verifica que 'yolov8n.pt' esté disponible.")
        st.session_state.is_running = False
    else:
        # Inicializar cámara
        if not st.session_state.camera_initialized:
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                st.error(f"❌ No se pudo abrir la cámara {camera_source}")
                st.session_state.is_running = False
            else:
                st.session_state.camera_initialized = True
                st.success(f"✅ Cámara {camera_source} inicializada correctamente")
        
        if st.session_state.camera_initialized:
            # Procesar frames en tiempo real
            stframe = video_placeholder.empty()
            
            # Configurar la cámara para mejor rendimiento
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Error al leer de la cámara")
                    break
                
                # Procesar frame
                processed_frame, person_count, total_equipment, occupied_stations, available_stations, monitor_count, laptop_count = process_frame(
                    frame, model, proximity_threshold, confidence_threshold)
                
                # Lógica de estabilización para los historiales
                st.session_state.monitor_history.append(total_equipment)
                st.session_state.occupied_history.append(occupied_stations)
                st.session_state.available_history.append(available_stations)
                
                # Calcular valores estables usando la moda
                try:
                    if st.session_state.monitor_history:
                        total_equipment_stable = statistics.mode(st.session_state.monitor_history)
                    else:
                        total_equipment_stable = 0
                except statistics.StatisticsError:
                    if st.session_state.monitor_history:
                        total_equipment_stable = st.session_state.monitor_history[-1]
                    else:
                        total_equipment_stable = 0
                
                try:
                    if st.session_state.occupied_history:
                        occupied_stable = statistics.mode(st.session_state.occupied_history)
                    else:
                        occupied_stable = 0
                except statistics.StatisticsError:
                    if st.session_state.occupied_history:
                        occupied_stable = st.session_state.occupied_history[-1]
                    else:
                        occupied_stable = 0
                
                try:
                    if st.session_state.available_history:
                        available_stable = statistics.mode(st.session_state.available_history)
                    else:
                        available_stable = 0
                except statistics.StatisticsError:
                    if st.session_state.available_history:
                        available_stable = st.session_state.available_history[-1]
                    else:
                        available_stable = 0
                
                # Añadir información al frame
                color_aforo = (0, 255, 0) if available_stable > 0 else (0, 0, 255)
                
                # Panel de información en el frame
                cv2.rectangle(processed_frame, (5, 5), (650, 140), (0, 0, 0), -1)
                cv2.putText(processed_frame, f"Puestos Libres: {available_stable}/{total_equipment_stable}", 
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color_aforo, 2)
                cv2.putText(processed_frame, f"Puestos Ocupados: {occupied_stable}", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(processed_frame, f"Personas: {person_count} | Monitores: {monitor_count} | Laptops: {laptop_count}", 
                           (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(processed_frame, f"Umbral proximidad: {proximity_threshold}px | Confianza: {confidence_threshold}", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Mostrar frame en Streamlit
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                
                # Actualizar interfaz cada 5 frames para mejor rendimiento
                frame_count += 1
                if frame_count % 5 == 0:
                    update_display(person_count, total_equipment_stable, occupied_stable, 
                                 available_stable, monitor_count, laptop_count)
                
                # Calcular y mostrar FPS y estadísticas
                if frame_count % 30 == 0:  # Cada segundo aproximadamente
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Calcular estadísticas de ocupación
                    avg_ocupacion = (occupied_stable / total_equipment_stable * 100) if total_equipment_stable > 0 else 0
                    
                    with chart_placeholder:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"📊 FPS: {fps:.1f}")
                            st.info(f"🎯 Frames: {frame_count}")
                        with col2:
                            st.info(f"📈 Ocupación Avg: {avg_ocupacion:.1f}%")
                            if total_equipment_stable > 0:
                                st.success(f"✅ {available_stable} de {total_equipment_stable} libres")
                            else:
                                st.warning("⚠️ No se detectaron equipos")
                
                # Pequeña pausa para evitar sobrecarga del CPU
                time.sleep(0.03)  # ~33 FPS máximo
            
            # Limpiar recursos
            if 'cap' in locals():
                cap.release()
                st.session_state.camera_initialized = False

# Información adicional en la parte inferior
st.markdown("---")
with st.expander("ℹ️ Información del Sistema"):
    st.markdown("""
    **Sistema de Control de Aforo Dinámico v3.0**
    
    - **Modelo de IA**: YOLOv8n (Ultralytics)
    - **Funcionalidades Mejoradas**:
        - ✅ Detección en tiempo real de personas, monitores y laptops
        - ✅ **Análisis inteligente de ocupación** basado en proximidad
        - ✅ Clasificación automática: Puesto Ocupado vs Puesto Libre
        - ✅ Contadores acumulativos de todas las detecciones
        - ✅ Estabilización basada en historial múltiple
        - ✅ Configuración dinámica de umbrales
        - ✅ Interfaz web moderna con métricas avanzadas
        
    - **Lógica de Ocupación**:
        - 🟢 **Puesto Libre**: Monitor/Laptop detectado SIN persona cerca
        - 🔴 **Puesto Ocupado**: Monitor/Laptop detectado CON persona dentro del umbral de proximidad
        - 👥 **Detección de Personas**: Se mantiene el conteo normal independiente
        
    - **Indicadores de Color**:
        - 🟢 **Verde**: Personas detectadas
        - 🔵 **Azul**: Equipos libres (monitor/laptop sin usar)
        - 🟠 **Naranja**: Equipos ocupados (monitor/laptop en uso)
        
    - **Controles Avanzados**:
        - 🎯 **Umbral de Proximidad**: Distancia para determinar si un equipo está ocupado
        - 🎯 **Umbral de Confianza**: Nivel mínimo para considerar detecciones válidas
        - 🔄 **Reset Contadores**: Reinicia contadores acumulativos
        - ⚙️ **Configuración de Historial**: Tamaño del buffer de estabilización
    """)

with st.expander("📊 Métricas y Estadísticas"):
    st.markdown("""
    **Panel de Métricas en Tiempo Real:**
    
    - **📋 Panel Principal**:
        - 🟢/🟡/🔴 **Estado General**: Puestos libres vs ocupados
        - 👥 **Personas Detectadas**: Conteo actual en frame
        - 🖥️💻 **Equipos**: Monitores y laptops detectados
        - 📊 **Ocupación**: Puestos ocupados vs libres
    
    - **📈 Métricas Sidebar**:
        - 📊 **Total Equipos**: Suma de monitores + laptops
        - 📈 **% Ocupación**: Porcentaje de equipos en uso
        - 👥 **Personas/Puesto**: Eficiencia de uso
    
    - **🔢 Contadores Totales**:
        - 📊 **Contadores Acumulativos**: Suma total desde inicio
        - 🔄 **Reseteable**: Pueden reiniciarse sin parar el sistema
        - 📈 **Historial**: Tracking completo de actividad
    
    **Algoritmo de Estabilización:**
    - Usa ventanas deslizantes para cada métrica
    - Calcula la moda (valor más frecuente) para estabilidad
    - Reduce falsos positivos por detecciones intermitentes
    - Mantiene precisión en condiciones variables
    """)

with st.expander("🔧 Instalación y Requisitos"):
    st.markdown("""
    **Dependencias necesarias:**
    ```bash
    pip install streamlit opencv-python ultralytics pillow numpy
    ```
    
    **Archivos requeridos:**
    - `yolov8n.pt`: Modelo YOLOv8 nano (se descarga automáticamente)
    - Cámara web conectada al sistema
    
    **Ejecutar la aplicación:**
    ```bash
    streamlit run control_aforo.py
    ```
    
    **Configuración Recomendada:**
    - 📹 **Cámara**: Resolución mínima 720p para mejor detección
    - 🖥️ **Hardware**: CPU moderna para procesamiento en tiempo real
    - 🌐 **Navegador**: Chrome/Firefox actualizados
    - 📡 **Red**: Conexión estable para video streaming
    
    **Optimización de Rendimiento:**
    - Ajusta el umbral de confianza según tu entorno
    - Reduce el tamaño del historial si necesitas respuesta más rápida
    - Usa cámara con buena iluminación para mejores detecciones
    """)

with st.expander("🎯 Casos de Uso y Aplicaciones"):
    st.markdown("""
    **Aplicaciones Principales:**
    
    - 🏢 **Oficinas Corporativas**:
        - Control de ocupación en espacios de trabajo
        - Optimización de recursos tecnológicos
        - Análisis de patrones de uso
    
    - 📚 **Bibliotecas y Centros de Estudio**:
        - Gestión de puestos de computadoras
        - Control de aforo en salas de estudio
        - Estadísticas de uso de equipos
    
    - 🎓 **Laboratorios Educativos**:
        - Monitoreo de estaciones de trabajo
        - Control de acceso a recursos limitados
        - Análisis de utilización de equipos
    
    - 🏬 **Espacios de Coworking**:
        - Gestión dinámica de puestos
        - Optimización de espacios compartidos
        - Métricas de ocupación para pricing
    
    **Beneficios Clave:**
    - ✅ **Automatización completa** sin intervención manual
    - ✅ **Precisión alta** con algoritmos de estabilización
    - ✅ **Tiempo real** con latencia mínima
    - ✅ **Escalabilidad** para múltiples cámaras
    - ✅ **Análisis avanzado** con métricas detalladas
    """)