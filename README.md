# Control de Aforo

Visión por computadora en tiempo real para saber cuántos puestos de trabajo están libres en un laboratorio, aula u oficina. Una cámara apunta al espacio, YOLOv8 detecta personas, monitores y laptops, y el sistema decide qué puestos están ocupados según la distancia entre cada persona y cada equipo.

Proyecto en equipo para la materia de Inteligencia Artificial, Ingeniería en Tecnologías de la Información, UIDE.

## Cómo funciona

1. **Detección.** Cada cuadro de la cámara pasa por YOLOv8. Se quedan las personas, los monitores (la clase `tv` del modelo) y las laptops que superan el umbral de confianza.
2. **Ocupación.** Un equipo está ocupado si el centro de alguna persona queda a menos de cierta distancia del centro del equipo. El umbral se ajusta en píxeles desde la barra lateral.
3. **Estabilización.** Las detecciones parpadean de un cuadro a otro. Por eso el conteo que se muestra es la moda de los últimos N cuadros, y no el valor de un solo cuadro.
4. **Panel.** Streamlit muestra el video con las cajas (verde para personas, azul para puestos libres, naranja para ocupados) y las métricas: puestos libres y ocupados, porcentaje de ocupación, personas por puesto y FPS.

## Instalación

Necesitas Python 3.9 o superior y una cámara web.

```bash
git clone https://github.com/rslcia11/controlForo.git
cd controlForo
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
streamlit run control_aforo.py
```

Se abre en el navegador. En la barra lateral eliges la cámara, ajustas los umbrales de proximidad y confianza, y pulsas **Iniciar**.

| Ajuste | Qué controla | Por defecto |
| :-- | :-- | :-- |
| Umbral de proximidad | Distancia máxima entre persona y equipo para contar el puesto como ocupado | 100 px |
| Umbral de confianza | Confianza mínima para aceptar una detección | 0.5 |
| Tamaño del historial | Cuadros usados para estabilizar el conteo | 30 |

## Dataset y modelo propio

Además del modelo general, entrenamos un detector propio de monitores:

- **Dataset:** 467 imágenes etiquetadas en Roboflow con las clases `monitor`, `laptop` y `pc`, divididas en 330 de entrenamiento, 91 de validación y 46 de prueba. Está en [`dataset/`](dataset) y publicado en [Roboflow Universe](https://universe.roboflow.com/controlaforo/monitordetection-buevj-x3xtm/dataset/1) con licencia CC BY 4.0.
- **Entrenamiento:** YOLOv8 con Ultralytics, 50 épocas.
- **Resultados en validación** para la clase `monitor`:

| Precisión | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :-: | :-: | :-: | :-: |
| 0.955 | 0.91 | 0.955 | 0.831 |

La app usa por defecto `yolov8n.pt`, el modelo preentrenado de Ultralytics, que ya reconoce personas, pantallas y laptops. Para probar el modelo propio, entrena con `dataset/data.yaml` y cambia la ruta en `load_model()`:

```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50
```

## Estructura

```
control_aforo.py   App de Streamlit: detección, ocupación y panel
yolov8n.pt         Pesos preentrenados de YOLOv8 nano
dataset/           Imágenes y etiquetas en formato YOLO, con data.yaml
```

## Stack

Python · Ultralytics YOLOv8 · OpenCV · Streamlit · Roboflow
