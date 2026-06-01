[`trt-sahi-yolo`](https://github.com/leon0514/trt-sahi-yolo)

Üstteki repo gerekli. Bu repoyu kullanmak için aşağıdaki yönergeyi oku.

Desteklenen modeller:
- YOLOv5
- YOLOv8
- YOLO11
- YOLO11-Pose
- YOLO11-Segmentation
- YOLO11-OBB
- D-FINE

Gereksinimler:
- CUDA
- TensorRT
- OpenCV
- Python
- Freetype

## 1. Makefile Ayarları

`Makefile` dosyasındaki aşağıdaki yolları kendi sisteminize göre değiştir:

```makefile
cuda_home := /usr/local/cuda
opencv_include_path := /path/to/opencv/include
opencv_library_path := /path/to/opencv/lib
trt_include_path := /path/to/tensorrt/include
trt_library_path := /path/to/tensorrt/lib
python_include_path := /path/to/python/include
```

TensorRT sürümünü seç:

```makefile
TRT_VERSION := 10
```

veya

```makefile
TRT_VERSION := 8
```

## 2. Projeyi Derle

Makefile ayarları tamamlandıktan sonra projeyi derle:

```bash
make
```

Derleme sonucunda Python tarafında kullanılacak paylaşımlı kütüphane oluşturulacak. Bunu inference scriptlerinin olduğu klasöre koy.

```text
workspace/trtsahi.so
```


## 3. YOLO Modelini ONNX Formatına Çevir

TensorRT engine oluşturmak için önce YOLO modelinin ONNX formatına çevir. Önemli noktalar:

- ONNX model **dynamic batch** desteklemeli. (Aksi takdirde SAHI görüntüyü fazla parçaya bölerse hata verebilir)
- `nms=False` olmalı. (NMS modelin içinde değil, repo tarafında yapılmalı)
- TensorRT tarafında NMS repo tarafından yapılır.
- YOLOv8 / YOLO11 çıktısı bazı durumlarda `1x84x8400` formatında olabilir. Repo, bu çıktının `1x8400x84` formatına uygun olmasını bekleyebilir.

  Örnek Ultralytics export:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.export(
    format="onnx",
    dynamic=True,
    simplify=True,
    opset=17,
    nms=False
)
```

## 4. YOLOv8 ve YOLOv11 için ONNX Modelini Transpose Et

trt-sahi-yolo/workspace içindeki v8trans.py scriptini çalıştır ve sonraki adımda bu scriptin verdiği ONNX modelini kullan.

```bash
python v8trans.py /path/to/model.onnx
```

## 5. Transposed ONNX Modelini TensorRT Engine Formatına Çevir

`trtexec` kullan. `maxShapes` içindeki batch değeri, SAHI sonucunda oluşacak parça sayısını karşılayacak kadar büyük olmalı. 
Python inference tarafında verilen `max_batch_size` değeri de bu ayarla uyumlu olmalı.

Örnek:

```bash
trtexec \
  --onnx=models/onnx/model.onnx \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:8x3x640x640 \
  --maxShapes=images:32x3x640x640 \
  --saveEngine=models/engine/model.engine \
  --fp16
```

## 6. Python ile Kullanım ve Temel Parametreler

Örnek kullanım:

```python
import cv2
import trtsahi

names = ["person", "helmet"]

model = trtsahi.TrtSahi(
    model_path="models/engine/helmet.engine",
    model_type=trtsahi.ModelType.YOLOV5SAHI,
    names=names,
    gpu_id=0,
    confidence_threshold=0.5,
    nms_threshold=0.4,
    max_batch_size=32,
    auto_slice=True,
    slice_width=640,
    slice_height=640,
    slice_horizontal_ratio=0.5,
    slice_vertical_ratio=0.5
)

images = [
    cv2.imread("inference/persons.jpg")
]

results = model.forwards(images)

for result in results[0]:
    print(result.box)
```

### `model_path`

TensorRT engine dosyasının yoludur.

Örnek:

```python
model_path="models/engine/model.engine"
```

### `model_type`

Kullanılan model tipini belirtir.

Örnekler:

```python
trtsahi.ModelType.YOLOV5SAHI
trtsahi.ModelType.YOLO11
trtsahi.ModelType.YOLO11POSESAHI
trtsahi.ModelType.YOLO11SEGSAHI
trtsahi.ModelType.YOLO11OBBSAHI
```

### `names`

Sınıf isimlerini içeren listedir.

Örnek:

```python
names = ["person", "helmet"]
```

### `gpu_id`

Kullanılacak GPU numarasıdır.

```python
gpu_id=0
```

### `confidence_threshold`

Minimum güven skoru eşiğidir.

```python
confidence_threshold=0.5
```

### `nms_threshold`

NMS eşiğidir.

```python
nms_threshold=0.4
```

### `max_batch_size`

Aynı anda işlenecek maksimum parça sayısıdır.

```python
max_batch_size=32
```

Bu değer, TensorRT engine oluştururken verilen `maxShapes` batch değeriyle uyumlu olmalıdır.

### `auto_slice`

SAHI benzeri otomatik parçalama işlemini açar veya kapatır.

```python
auto_slice=True
```

### `slice_width` ve `slice_height`

Her parçanın genişlik ve yükseklik değeridir.

```python
slice_width=640
slice_height=640
```

### `slice_horizontal_ratio` ve `slice_vertical_ratio`

Parçalar arasındaki yatay ve dikey overlap oranıdır.

```python
slice_horizontal_ratio=0.5
slice_vertical_ratio=0.5
```

Overlap arttıkça sınırdaki nesnelerin kaçırılma ihtimali azalır, ancak işlem maliyeti artar.

