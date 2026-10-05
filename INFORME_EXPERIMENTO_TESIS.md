# Informe del experimento de clasificación y detección de granos de café

**Versión de trabajo para integrar y adaptar a la tesis**  
**Fecha de corte:** 2 de octubre de 2026  
**Repositorio:** `navarrete_tesis`

> Este informe describe lo implementado y ejecutado en el repositorio y en la sesión de trabajo. Distingue los resultados medidos de los componentes que solo están preparados o propuestos. Antes de incorporarlo como capítulo final, se deben añadir las referencias bibliográficas utilizadas en la investigación y confirmar las reglas de etiquetado con el protocolo de la tesis.

## 1. Resumen

Se organizó un flujo experimental para clasificar granos individuales de café verde en las categorías `sano` y `defecto`, utilizando imágenes de la partición `train` de USK-Coffee. Las imágenes se dividieron en conjuntos de entrenamiento, validación y prueba con una proporción 80/10/10 y semilla 42. Las particiones resultantes contienen 1.920 imágenes de entrenamiento, 240 de validación y 240 de prueba, equilibradas entre las dos clases.

Se prepararon y entrenaron tres alternativas: YOLOv8-cls, una Slim-CNN compacta y una ResNet18 siamesa. El aumento de datos se aplicó en línea a las imágenes de entrenamiento. Los entrenamientos tuvieron un máximo de 100 épocas y utilizaron parada temprana; se conservaron los checkpoints y los historiales. En el conjunto de prueba, YOLOv8-cls obtuvo 96,67 % de exactitud top-1, Slim-CNN 82,92 % y la siamesa 85 % de exactitud de verificación de pares. Esta última mide una tarea distinta y no es directamente comparable con la clasificación de imágenes.

El resultado de YOLOv8-cls corresponde a clasificación de imágenes individuales, no a detección por cajas. También se implementó un prototipo híbrido OpenCV + Slim-CNN para proponer regiones en imágenes con varios granos y clasificar cada recorte. La línea de YOLOv8-detect está preparada en código, pero no se entrenó porque todavía no existe un conjunto revisado de cajas por grano. Por lo tanto, el experimento permite concluir sobre clasificación en USK-Coffee, pero no demostrar aún detección fiable en escenas con múltiples granos.

## 2. Objetivo y alcance

El objetivo experimental ejecutado fue comparar modelos que clasifican una imagen de un grano como `sano` o `defecto`, manteniendo particiones comunes de entrenamiento, validación y prueba.

En paralelo, se definió una extensión para analizar imágenes con varios granos. Se separaron dos problemas:

1. **Clasificación individual:** determinar la clase de una imagen que contiene un grano. Esta tarea sí cuenta con etiquetas por imagen en USK-Coffee.
2. **Detección de instancias:** localizar cada grano dentro de una escena y asignarle una clase. Esta tarea requiere cajas por instancia y evaluación de localización, que aún no se han preparado para escenas múltiples.

En consecuencia, YOLOv8-cls no se considera un detector y su exactitud no se presenta como una métrica de localización. La detección híbrida OpenCV + clasificador y el detector YOLO son líneas de desarrollo posteriores al experimento de clasificación.

## 3. Datos y preparación

### 3.0 Evolución de la selección de datos

Durante la exploración inicial se mostró una partición preliminar con 3.720 imágenes de entrenamiento, 401 de validación y 462 de prueba, asociada a una selección más amplia de datos. Después se decidió usar exclusivamente las imágenes de `USK-Coffee/train`; se regeneró el índice y se sustituyó esa partición preliminar por la división final de 2.400 imágenes descrita aquí. Los conteos preliminares no forman parte de los resultados reportados en este informe.

### 3.1 Fuente y etiquetas

Se utilizaron exclusivamente imágenes de:

```text
data/raw/USK-Coffee/train/
```

El conjunto se leyó desde las carpetas originales `premium/` y `defect/`, que el proyecto mapeó, respectivamente, a `sano` y `defecto`. No se incorporaron al experimento de clasificación imágenes de `Specific-Defect` ni los recortes provenientes de sus hojas de contacto. La decisión de trabajar solo con USK-Coffee evita mezclar una etiqueta por grano con etiquetas originalmente asignadas a una hoja completa.

Las imágenes de USK-Coffee contienen un grano individual centrado sobre un fondo claro. Por ello, las etiquetas disponibles son adecuadas para clasificación individual; no contienen anotaciones originales de caja.

Las particiones `val` y `test` originales del dataset fuente no se incorporaron al experimento; la validación y prueba aquí descritas se derivaron de `USK-Coffee/train`, de acuerdo con la decisión metodológica tomada durante el trabajo.

### 3.2 División de datos

El generador `src/data/build_splits.py` reunió las imágenes de las dos carpetas, asignó la clase según su carpeta y generó particiones con proporciones 80 % / 10 % / 10 %, estratificadas por clase cuando fue posible y con semilla 42. Cada archivo fuente se trató como una unidad independiente y se registró en `data/splits/index.csv`, junto con su ruta original, clase y partición.

| Partición | Sano | Defecto | Total |
|---|---:|---:|---:|
| Entrenamiento | 960 | 960 | 1.920 |
| Validación | 120 | 120 | 240 |
| Prueba | 120 | 120 | 240 |
| **Total** | **1.200** | **1.200** | **2.400** |

La partición de prueba permaneció separada durante el ajuste de épocas y la selección de checkpoints. Posteriormente se ejecutó una evaluación final sobre ella para los tres modelos.

La separación evita reutilizar el mismo archivo fuente en varias particiones. Sin embargo, no equivale a una separación por finca, cámara o sesión de captura: el experimento no contó con esos grupos de procedencia. Las conclusiones se restringen, por tanto, a la distribución representada por USK-Coffee.

## 4. Preprocesamiento y aumento de datos

Las rutas de clasificación utilizan imágenes redimensionadas a 224 × 224 píxeles. Slim-CNN y ResNet18 siamesa convierten las imágenes a tensores y normalizan los canales con media `[0.485, 0.456, 0.406]` y desviación estándar `[0.229, 0.224, 0.225]`.

El aumento se implementó en línea: no se crearon duplicados permanentes. En cada presentación al modelo podían aplicarse reflejos horizontal y vertical, rotación de hasta 180°, pequeños desplazamientos y cambios moderados de escala, brillo, contraste, saturación y tono. Solo se aplicó a entrenamiento; validación y prueba conservaron imágenes sin aumento. La política se registra en `config/augmentation.yaml` y estaba habilitada durante las corridas descritas.

El número de archivos no aumentó por esta política. Con 1.920 imágenes y lote de 32, Slim-CNN tuvo 60 iteraciones de optimización por época. Las variaciones se generaron al presentar esas imágenes durante las épocas.

YOLOv8-cls recibió la configuración de aumento mediante sus propios parámetros de Ultralytics (reflejos, escala y cambios HSV, entre otros). No todas las operaciones de la política PyTorch se transfieren de forma idéntica a YOLO. Por ello, estas corridas comparan modelos con sus pipelines actuales; no constituyen una ablación controlada que aísle el efecto causal del aumento. No se entrenó una segunda versión de cada modelo con `enabled: false`.

## 5. Modelos y procedimiento de entrenamiento

Los tres clasificadores utilizaron las mismas particiones de imágenes individuales. El máximo configurado fue de 100 épocas. Slim-CNN y la siamesa utilizaron parada temprana por pérdida de validación, con paciencia de 15 épocas y `min_delta=0.0001`. Se guardaron `best.pt`, `last.pt`, `history.csv` y `history.json`; `best.pt` conserva el checkpoint seleccionado por validación y `last.pt` el último estado para reanudar.

### 5.1 YOLOv8-cls

Se utilizó `yolov8n-cls.pt` preentrenado, con imágenes de 224 × 224 y lote de 32. El modelo recibe una imagen completa y devuelve una clase; el entrenamiento no emplea cajas. La parada temprana de Ultralytics se configuró con paciencia de 15. Los resultados se guardaron en una carpeta de corrida independiente bajo `checkpoints/`; para la corrida evaluada se utilizó `checkpoints/yolov8_cls-2/weights/best.pt`.

### 5.2 Slim-CNN

Se entrenó una CNN compacta desarrollada en el repositorio, con bloques convolucionales separables, canales `[24, 48, 96, 160]` y dropout de 0,25. Se utilizó AdamW con tasa de aprendizaje inicial 0,001, lote de 32 y semilla 42. Un planificador `ReduceLROnPlateau` redujo la tasa cuando `val_loss` se estancó. La selección del mejor checkpoint se basó en la menor pérdida de validación.

La implementación debe describirse como la Slim-CNN de este repositorio. No se afirma que replique exactamente una arquitectura publicada sin contrastar sus capas y dimensiones con la referencia bibliográfica específica.

### 5.3 ResNet18 siamesa

Se implementó una ResNet18 compartida entre dos ramas, entrenada con pérdida contrastiva y margen 1,0. La configuración utilizó dimensión de embedding 128, imágenes de 224 × 224, lote de 32 y AdamW con tasa inicial 0,0003. El entrenamiento partió sin pesos preentrenados de ImageNet.

El generador `src/data/generate_pairs.py` formó pares positivos con dos imágenes de la misma clase y pares negativos con imágenes de clases diferentes, equilibrando la cantidad de ambos tipos dentro de cada partición. Los pares se generaron dentro de train, validación y test por separado. Para la evaluación de prueba se utilizaron 240 pares: 120 `same` y 120 `different`.

La salida de este modelo es una distancia entre embeddings. El umbral de decisión se estimó con los pares de validación y se guardó en el checkpoint; no se ajustó con los pares de test.

## 6. Entorno computacional

El entrenamiento se ejecutó en la máquina remota de laboratorio `razorcrest`. La comprobación por SSH identificó un AMD Ryzen Threadripper PRO 5955WX (16 núcleos físicos, 32 hilos), 251 GiB de memoria y dos GPU NVIDIA RTX A6000 de 48 GiB cada una. Se reportaron aproximadamente 449 GB libres en el volumen del proyecto.

Inicialmente, el entorno tenía PyTorch compilado para CUDA 13.0 y el controlador instalado ofrecía CUDA 12.8, por lo que `torch.cuda.is_available()` devolvía falso. Se instaló PyTorch 2.11.0+cu128 en el entorno virtual del proyecto. La verificación posterior indicó CUDA disponible y dos GPU visibles. La salida de evaluación de YOLO confirmó ejecución en `CUDA:0 (NVIDIA RTX A6000)`.

## 7. Resultados

### 7.1 Historial de entrenamiento y parada temprana

| Modelo | Épocas ejecutadas | Iteraciones de entrenamiento | Mejor resultado de validación registrado |
|---|---:|---:|---|
| Slim-CNN | 46 de 100 | 2.760 | Época 31: `val_loss=0.3804`, `val_accuracy=84,58 %` |
| YOLOv8-cls | 26 de 100 | Aproximadamente 1.560 batches (60 por época, calculado desde 1.920 imágenes y lote 32) | Época 11: `top1_acc=97,92 %`; último valor de validación: 95,42 % |
| ResNet18 siamesa | 80 de 100 | 4.800 batches de pares | Época 65: `val_loss=0.1099`, `val_pair_accuracy=88,75 %` |

Los entrenamientos terminaron antes del límite máximo. Para Slim-CNN y la siamesa, la selección de `best.pt` se basó en `val_loss`; por eso la época con la mayor exactitud de validación no necesariamente coincide con el checkpoint seleccionado. En la siamesa, por ejemplo, la exactitud máxima de pares registrada fue 89,58 % en la época 61, mientras que el checkpoint escogido por menor pérdida correspondió a la época 65.

### 7.2 Evaluación final en test

| Modelo | Métrica principal | Resultado | Interpretación |
|---|---|---:|---|
| YOLOv8-cls | Exactitud top-1 | **96,67 %** (232/240) | Clasificación de una imagen individual en dos clases. |
| Slim-CNN | Exactitud | **82,92 %** (199/240) | Clasificación de una imagen individual en dos clases. |
| ResNet18 siamesa | Exactitud de pares | **85,00 %** (204/240 pares) | Decisión `same`/`different`; tarea diferente a clasificar una imagen como sana o defectuosa. |

**YOLOv8-cls.** La matriz de confusión registró 116 aciertos de 120 para cada clase: cuatro granos defectuosos se predijeron como sanos y cuatro sanos como defectuosos. El resultado `top5_acc=100 %` no es informativo en una tarea de dos clases, porque ambas clases caben en las cinco opciones. La interpretación se centra en `top1_acc`.

**Slim-CNN.** La exactitud fue 82,92 %, con macro F1 de 82,92 %. En el orden de clases `sano`, `defecto`, la matriz fue `[[100, 20], [21, 99]]`: 20 granos sanos se clasificaron como defectuosos y 21 defectuosos como sanos. Las precisiones por clase fueron 82,64 % para sano y 83,19 % para defecto.

**ResNet18 siamesa.** La exactitud de verificación fue 85 %. En el orden `different`, `same`, la matriz fue `[[100, 20], [16, 104]]`. Hubo 20 pares de clases diferentes predichos como iguales y 16 pares iguales predichos como diferentes. Este porcentaje no debe ordenarse en la misma tabla comparativa de exactitud por grano como si midiera la misma tarea.

## 8. Interpretación

En la tarea de clasificación individual y bajo las particiones internas de USK-Coffee, YOLOv8-cls obtuvo el mejor resultado observado. Su prueba top-1 fue 96,67 %, frente a 82,92 % de Slim-CNN. La matriz de confusión de YOLO muestra errores similares entre ambas clases, lo que sugiere que el resultado no proviene de acertar únicamente la clase mayoritaria; de hecho, el test estuvo balanceado.

La ResNet18 siamesa mostró capacidad para discriminar pares, pero su métrica describe verificación y no clasificación binaria de una imagen. Para compararla con los otros clasificadores habría que definir una regla adicional —por ejemplo, comparación contra referencias o centroides— y evaluarla como clasificador en el mismo conjunto y bajo un protocolo establecido antes de medir.

Los resultados describen el desempeño en imágenes individuales procedentes de la misma fuente y distribución. No prueban robustez frente a otras cámaras, fondos, fincas, condiciones de luz o escenas con granos superpuestos. Tampoco permiten atribuir la diferencia entre arquitecturas al aumento de datos, porque no se ejecutó un par controlado con y sin aumento para cada modelo.

## 9. Desarrollo de la detección

El cuaderno `05_deteccion_hibrida.ipynb` y `src/inference/multi_bean.py` implementan una propuesta híbrida: OpenCV estima el fondo desde los bordes, segmenta el primer plano por diferencia de color en Lab y umbralización, aplica operaciones morfológicas, busca contornos externos y genera cajas candidatas. Cada recorte se clasifica con un checkpoint Slim-CNN. Es una localización clásica seguida de clasificación, no un detector neuronal.

El prototipo supone un fondo claro y granos suficientemente separados. Si varios granos se tocan, OpenCV puede producir una sola región; sombras y bajo contraste también pueden afectar la segmentación. La implementación requiere una fotografía representativa de varios granos para ser probada visualmente. No se registró una evaluación cuantitativa del pipeline híbrido sobre un conjunto anotado de escenas múltiples.

También se incorporaron scripts y configuración para YOLOv8-detect. `config/yolo_detect.yaml` establece dos clases y un máximo de 100 épocas, pero el código exige cajas revisadas en `data/detection/images/{train,val,test}` y `data/detection/labels/{train,val,test}`. Esa estructura de anotaciones no estaba disponible, por lo que no se ejecutó el detector ni se reporta mAP.

OpenCV puede acelerar la creación de etiquetas preliminares para fotos de fondo claro, pero sus cajas deben revisarse y sus clases deben asignarse o validarse manualmente. Las imágenes individuales de USK-Coffee podrían aportar una caja ajustada por grano y heredar la etiqueta de la carpeta; no sustituyen una prueba en fotos con varios granos. Para sostener resultados de detección se requiere recopilar escenas múltiples, anotar cada instancia y reservar escenas reales independientes para validación y test.

## 10. Limitaciones y validez

1. **Una sola partición aleatoria:** se usó una semilla y un reparto 80/10/10. No se repitió el experimento con varias semillas ni con validación cruzada.
2. **Prueba interna:** las 240 imágenes de test provienen de USK-Coffee y comparten su distribución de captura. Falta un conjunto externo con imágenes de otros entornos.
3. **Calidad de etiquetas:** las clases se heredaron de las carpetas fuente. No se hizo una auditoría experta completa de posibles casos visualmente ambiguos.
4. **Aumento no aislado:** no se entrenaron versiones pareadas con aumento desactivado. Por ello no hay una estimación experimental del efecto del aumento.
5. **Pipelines de aumento distintos:** el aumento usado por PyTorch y el mapeo de operaciones a Ultralytics no son idénticos.
6. **Tarea siamesa diferente:** sus pares no son unidades de observación independientes equivalentes a 240 imágenes clasificadas; una imagen puede intervenir en más de un par negativo.
7. **Detección pendiente:** no hay ground truth de cajas en escenas múltiples ni métricas de localización. Los resultados de YOLOv8-cls no constituyen evidencia de detección.
8. **Arquitectura Slim-CNN:** la red es una implementación compacta propia y debe cotejarse con la publicación específica antes de afirmar reproducción exacta de un método bibliográfico.

Como las métricas de test ya se consultaron, cualquier ajuste posterior motivado por esos errores debe validarse con un nuevo conjunto independiente o con un protocolo de evaluación adicional. No se deben volver a presentar las mismas 240 imágenes como una prueba intacta después de optimizar usando sus resultados.

## 11. Plan de trabajo posterior

1. **Cerrar el informe de clasificación:** conservar checkpoints, configuraciones, `index.csv` e informes JSON; añadir las referencias bibliográficas y documentar explícitamente que YOLO evaluado fue YOLOv8-cls.
2. **Realizar análisis descriptivo de errores:** inspeccionar los errores y las matrices de confusión para localizar posibles ambigüedades. Usar estos hallazgos para plantear experimentos, no para volver a optimizar contra el mismo test.
3. **Probar OpenCV + Slim-CNN:** reunir fotografías con varios granos, ejecutar el cuaderno 05 y revisar si la segmentación produce una caja por grano. Separar fallos de localización de fallos de clasificación.
4. **Crear un conjunto de detección:** recopilar escenas variadas, generar propuestas de caja con OpenCV y corregir manualmente caja y clase. Asignar particiones por escena o sesión de captura.
5. **Entrenar YOLOv8-detect:** solo después de validar las etiquetas y asegurar representación de clases en cada partición. Reportar precisión/recall y mAP con las escenas reservadas.
6. **Comparar detectores en la misma prueba:** medir YOLOv8-detect y el pipeline híbrido sobre las mismas escenas, con métricas de localización, clase y tiempo de inferencia.
7. **Medir adaptación:** reservar fotos externas con cámaras, iluminación y fondos distintos. El aumento sintético en imágenes de fondo blanco no reemplaza estos datos.

## 12. Archivos para reproducibilidad y trazabilidad

- Particiones y auditoría de procedencia: `data/splits/` y `data/splits/index.csv`.
- Configuración de datos: `config/dataset_config.yaml`.
- Configuración de aumento: `config/augmentation.yaml`.
- Configuración de modelos: `config/slim_cnn.yaml`, `config/yolo.yaml`, `config/siamese.yaml`.
- Checkpoints e historiales: `checkpoints/slim_cnn/`, `checkpoints/yolov8_cls-2/` y `checkpoints/siamese_resnet18/`.
- Métricas de test: `reports/slim_cnn_test.json`, `reports/yolov8_cls_test_metrics.json`, `reports/yolov8_cls_test/` y `reports/siamese_resnet18_test.json`.
- Preparación y exploración: `notebooks/01_inspeccion_crops.ipynb`, `notebooks/02_exploracion_dataset.ipynb`.
- Aumento y entrenamiento: `notebooks/03_aumento_datos.ipynb`, `notebooks/04_entrenamiento_modelos.ipynb`.
- Prototipo híbrido: `notebooks/05_deteccion_hibrida.ipynb` y `src/inference/multi_bean.py`.

Para repetir evaluación de clasificación desde la raíz del proyecto y el entorno virtual:

```bash
python -m src.evaluate_all --model slim_cnn
python -m src.evaluate_all --model yolov8_cls --checkpoint checkpoints/yolov8_cls-2/weights/best.pt
python -m src.evaluate_all --model siamese_resnet18
```
