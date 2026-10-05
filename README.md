# Visión artificial para granos de café

El flujo de clasificación usa únicamente las imágenes originales de `data/raw/USK-Coffee/train/`. El notebook `02_exploracion_dataset.ipynb` crea particiones internas train/val/test. No usa las carpetas originales `val` y `test` ni los recortes de `Specific-Defect`.

## Entorno

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m ipykernel install --user --name tesis-granos-verdes --display-name "Python (tesis-granos-verdes)"
```

CUDA no es obligatorio; los modelos pueden ejecutarse en CPU, con mayor tiempo de entrenamiento. YOLOv8-cls usa pesos preentrenados por defecto y podría descargarlos en su primera ejecución.

## Datos de clasificación

```text
data/raw/USK-Coffee/train/
├── defect/    # mapeado a defecto
└── premium/   # mapeado a sano
```

Las carpetas fuente se configuran en `config/dataset_config.yaml`. Ejecuta `02_exploracion_dataset.ipynb` para crear `data/splits/{train,val,test}/{sano,defecto}` e inspeccionar `data/splits/index.csv`.

## Aumento de datos

Abre primero `03_aumento_datos.ipynb`. Muestra ejemplos de las transformaciones y documenta su alcance. Los aumentos se aplican aleatoriamente durante el entrenamiento, solo a las imágenes de train; no crean copias en disco y no alteran validación ni test. La política se configura en `config/augmentation.yaml` y se puede desactivar con `enabled: false` para hacer una comparación controlada.

Se proponen rotaciones y reflejos por la orientación variable de los granos, pequeños cambios de encuadre y cambios suaves de iluminación/color. Esto puede mejorar robustez, pero no garantiza mayor precisión. Si train y test tienen fondo blanco, estos aumentos no demuestran que el modelo se adapte a fondos o cámaras distintos; para eso se requieren imágenes reales de esos entornos, reservadas como evaluación externa.

Para comparar Slim-CNN con y sin aumento sin sobrescribir checkpoints, cambia `enabled` en ese YAML entre corridas y asigna otra salida, por ejemplo `python -m src.train --model slim_cnn --output checkpoints/slim_cnn_sin_aumento` y luego `python -m src.train --model slim_cnn --output checkpoints/slim_cnn_con_aumento`. En YOLO usa `--run-name yolov8_cls_sin_aumento` y `--run-name yolov8_cls_con_aumento`. Mantén las mismas particiones, semilla y número de épocas; decide con validación y reserva test para el resultado final.

## Entrenar clasificadores

Abre `04_entrenamiento_modelos.ipynb`, selecciona el kernel `Python (tesis-granos-verdes)`, ejecuta la comprobación inicial y luego **solo la celda del modelo deseado**:

- **YOLOv8-cls:** clasificación de imagen completa por carpeta. `python -m src.train --model yolov8_cls`. Configuración: `config/yolo.yaml`; Ultralytics guarda `best.pt`, `last.pt` y resultados en `checkpoints/yolov8_cls/` (o en la siguiente carpeta numerada si ya existe).
- **Slim-CNN:** CNN compacta con convoluciones separables. `python -m src.train --model slim_cnn`. Configuración: `config/slim_cnn.yaml`; resultados en `checkpoints/slim_cnn/`.
- **ResNet18 siamesa:** encoder compartido para pares positivos/negativos y pérdida contrastiva. La celda del notebook genera pares; también se puede ejecutar `python -m src.data.generate_pairs`. Configuración: `config/siamese.yaml`; resultados en `checkpoints/siamese_resnet18/`.

Los tres clasificadores tienen un límite de 100 épocas. La parada temprana usa la pérdida de validación (`val_loss` en Slim-CNN y la siamesa; Ultralytics administra YOLO), paciencia 15 y no usa el conjunto test. Slim-CNN y la siamesa guardan `best.pt` (mejor validación), `last.pt` (último estado con optimizador, scheduler y contador de parada), `history.csv` y `history.json`. `last.pt` se escribe al final de cada época para poder reanudar después de una interrupción. Las salidas existentes no se sobrescriben en los entrenadores propios: especifica una nueva ruta `--output` para otra corrida. Para continuar una corrida propia, usa `python -m src.train --model slim_cnn --resume` o `python -m src.train --model siamese_resnet18 --resume`, conservando la misma carpeta `--output`. Para YOLO, usa `--resume ruta/a/weights/last.pt`.

Con 1,920 imágenes de entrenamiento y batch 32, Slim-CNN ejecuta 60 iteraciones de optimización por época: hasta 6,000 si alcanza las 100 épocas. El registro muestra iteraciones, paso global, pérdidas, tasa de aprendizaje y tiempo por época.

La Slim-CNN implementada es una baseline compacta; debe cotejarse con la referencia concreta de la bibliografía para afirmar que replica exactamente sus capas y dimensiones.

## Prototipo de múltiples granos con OpenCV + Slim-CNN

Para fotos claras donde los granos no se tocan, se puede localizar con OpenCV y clasificar cada recorte con Slim-CNN. Copia una foto de escena a `data/inference_examples/` o cambia `IMAGE_PATH` en `05_deteccion_hibrida.ipynb`. El notebook puede previsualizar la máscara y las cajas candidatas de OpenCV antes de entrenar Slim-CNN; para ver las clases predichas, primero se requiere el checkpoint. El pipeline produce recortes y cajas coloreadas por clase. También está disponible como `python -m src.predict_multi_beans --source ruta/foto.jpg`; agrega `--only defecto` para visualizar solo las cajas defectuosas.

Esto es localización clásica seguida de clasificación, no detección neuronal. La segmentación actual usa la diferencia de color respecto al borde, operaciones morfológicas y contornos. Granos en contacto pueden quedar en una caja conjunta; sombras, bordes poco contrastados o fondos que no sean claros pueden generar fallas. Valídalo con fotos representativas antes de usarlo. El código no puede correr sin un checkpoint Slim-CNN entrenado.

## Detección de varios granos en una foto

La clasificación YOLOv8-cls no dibuja cajas. Para detectar una foto con varios granos, se incorporó un pipeline separado `yolov8_detect` y un validador de anotaciones. **Aún no se puede entrenar:** actualmente no hay etiquetas de caja por cada grano. La etiqueta por hoja de `Specific-Defect` no basta porque una hoja de defecto puede contener granos sanos; copiar la etiqueta a cada recorte produciría supervisión incorrecta.

Primero hay que anotar manualmente las escenas con cajas alrededor de todos los granos y etiquetas `sano` o `defecto`. Formato requerido:

```text
data/detection/
├── images/{train,val,test}/foto.jpg
└── labels/{train,val,test}/foto.txt
```

Cada línea del TXT: `class_id centro_x centro_y ancho alto`; coordenadas normalizadas de 0 a 1, con `0=sano`, `1=defecto`. Se necesita una caja completa por cada grano y archivos vacíos para fotos sin objetos. Asigna todas las fotos de una misma hoja/sesión a la misma partición para evitar fuga entre train y test. Usa escenas diversas; las 11 grillas `Specific-Defect` por sí solas son una base muy pequeña para generalizar.

Una vez anotado, ejecutar `python -m src.train --model yolov8_detect` entrena el detector (`config/yolo_detect.yaml`) y guarda `checkpoints/yolov8_detect/weights/best.pt`. El validador detiene el entrenamiento si faltan cajas, las coordenadas son inválidas o no están representadas las clases.

Con el detector entrenado, inferencia sobre una foto:

```bash
python -m src.predict_yolo_detect --source ruta/a/foto_con_varios_granos.jpg
```

Para mostrar solo granos defectuosos, agrega `--classes 1`. La evaluación del detector informa mAP y se ejecuta con `python -m src.evaluate_all --model yolov8_detect`.

## Evaluación de clasificación

Evalúa cada clasificador una vez en el split interno `test`:

```bash
python -m src.evaluate_all --model yolov8_cls
python -m src.evaluate_all --model slim_cnn
python -m src.evaluate_all --model siamese_resnet18
```

Las métricas se guardan bajo `reports/`. El test es una partición derivada de la carpeta fuente `train` de USK-Coffee, no el test oficial original.
