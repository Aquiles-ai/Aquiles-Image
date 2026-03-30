<div align="center">

# Aquiles-Image

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1763763684/aquiles_image_m6ej7u.png" alt="Aquiles-Image Logo" width="800"/>

### **Generación de imágenes/videos autoalojada con APIs compatibles con OpenAI**

*🚀 FastAPI • Diffusers • Reemplazo directo de OpenAI*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-orange.svg)](https://platform.openai.com/docs/api-reference/images)
[![PyPI Version](https://img.shields.io/pypi/v/aquiles-image.svg)](https://pypi.org/project/aquiles-image/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/aquiles-image?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=MAGENTA&left_text=downloads)](https://pypi.org/project/aquiles-image/)
[![Docs](https://img.shields.io/badge/Docs-Read%20the%20Docs-brightgreen.svg)](https://aquiles-ai.github.io/aquiles-image-docs/) 
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Aquiles-ai/Aquiles-Image)
[![View Code Wiki](https://www.gstatic.com/_/boq-sdlc-agents-ui/_/r/YUi5dj2UWvE.svg)](https://codewiki.google/github.com/aquiles-ai/aquiles-image)

**\[ [English](README.md) | Español \]**

</div>

## 🎯 ¿Qué es Aquiles-Image?

**Aquiles-Image** es un servidor de API listo para producción que te permite ejecutar modelos de generación de imágenes de última generación en tu propia infraestructura. Diseñado para ser compatible con OpenAI, puedes cambiar de servicios externos a autoalojado en menos de 5 minutos.

### ¿Por qué Aquiles-Image?

| Problema | Solución de Aquiles-Image |
|----------|---------------------------|
| 💸 **APIs externas costosas** | Ejecuta modelos localmente con uso ilimitado |
| 🔒 **Preocupaciones de privacidad** | Tus imágenes nunca salen de tu servidor |
| 🐌 **Inferencia lenta** | Optimizaciones avanzadas para generación 3x más rápida |
| 🔧 **Configuración compleja** | Un solo comando para ejecutar cualquier modelo soportado |
| 🚫 **Dependencia del proveedor** | Compatible con OpenAI, cambia sin reescribir tu código |

### Características Principales

- **🔌 Compatible con OpenAI** - Usa el cliente oficial de OpenAI sin ningún cambio en el código
- **⚡ Agrupación Inteligente** - Agrupación automática de solicitudes por parámetros compartidos para máximo rendimiento en configuraciones de una o múltiples GPUs
- **🎨 Más de 30 Modelos Optimizados** - 18 de imagen (FLUX, SD3.5, Qwen) + 12 de video (Wan2.x, HunyuanVideo) + ilimitados vía AutoPipeline (solo T2I)
- **🚀 Soporte Multi-GPU** - Inferencia distribuida con balanceo de carga dinámico entre GPUs (modelos de imagen) para escalado horizontal
- **🛠️ Excelente Experiencia de Desarrollo** - CLI simple, modo dev para pruebas, monitoreo integrado
- **🎬 Video Avanzado** - Texto a video con las series Wan2.x y HunyuanVideo (+ variantes Turbo)
- **🧩 Soporte de LoRA** - Carga cualquier LoRA desde HuggingFace o un path local mediante un archivo JSON de configuración, compatible con todos los modelos nativos y AutoPipeline

## 🚀 Inicio Rápido

### Instalación

```bash
# Desde PyPI (recomendado)
pip install aquiles-image

# Desde el código fuente
git clone https://github.com/Aquiles-ai/Aquiles-Image.git
cd Aquiles-Image
pip install .
```

### Iniciar el Servidor

**Modo de Dispositivo Único (Por defecto)**
```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium"
```

**Modo distribuido multi-GPU (solo para modelos de imagen)**
```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium" --dist-inference
```

> **Nota sobre Inferencia Distribuida**: Activa el modo multi-GPU agregando la flag `--dist-inference`. Cada GPU cargará una copia del modelo, así que asegúrate de que cada GPU tenga suficiente VRAM. El sistema balancea automáticamente la carga entre GPUs y agrupa solicitudes con parámetros compartidos para máximo rendimiento.

### Genera tu Primera Imagen

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5500", api_key="not-needed")

result = client.images.generate(
    model="stabilityai/stable-diffusion-3.5-medium",
    prompt="a white siamese cat",
    size="1024x1024"
)

print(f"Image URL: {result.data[0].url}")
```

¡Eso es todo! Ahora estás generando imágenes con la misma API que usarías con OpenAI.

## 🎨 Modelos soportados

### Texto a Imagen (`/images/generations`)

- `stabilityai/stable-diffusion-3-medium`
- `stabilityai/stable-diffusion-3.5-medium` 
- `stabilityai/stable-diffusion-3.5-large`
- `stabilityai/stable-diffusion-3.5-large-turbo`
- `black-forest-labs/FLUX.1-dev`
- `black-forest-labs/FLUX.1-schnell`
- `black-forest-labs/FLUX.1-Krea-dev`
- `black-forest-labs/FLUX.2-dev` * 
- `diffusers/FLUX.2-dev-bnb-4bit`
- `Tongyi-MAI/Z-Image-Turbo`
- `Qwen/Qwen-Image`
- `Qwen/Qwen-Image-2512`
- `black-forest-labs/FLUX.2-klein-4B`
- `black-forest-labs/FLUX.2-klein-9B`
- `zai-org/GLM-Image` - (Este modelo suele ser el más lento de ejecutar en términos relativos)
- `Tongyi-MAI/Z-Image`
- `black-forest-labs/FLUX.2-klein-9b-kv`

### Imagen a Imagen (`/images/edits`)

- `black-forest-labs/FLUX.1-Kontext-dev`
- `diffusers/FLUX.2-dev-bnb-4bit` - Soporta edición de múltiples imágenes. Máximo 10 imágenes de entrada.
- `black-forest-labs/FLUX.2-dev` * - Soporta edición de múltiples imágenes. Máximo 10 imágenes de entrada.
- `Qwen/Qwen-Image-Edit` 
- `Qwen/Qwen-Image-Edit-2509` - Soporta edición de múltiples imágenes. Máximo 3 imágenes de entrada.
- `Qwen/Qwen-Image-Edit-2511` - Soporta edición de múltiples imágenes. Máximo 3 imágenes de entrada.
- `black-forest-labs/FLUX.2-klein-4B` - Soporta edición de múltiples imágenes. Máximo 10 imágenes de entrada.
- `black-forest-labs/FLUX.2-klein-9B` - Soporta edición de múltiples imágenes. Máximo 10 imágenes de entrada.
- `black-forest-labs/FLUX.2-klein-9b-kv` - Soporta edición de múltiples imágenes. Máximo 10 imágenes de entrada.
- `zai-org/GLM-Image` - Soporta edición de múltiples imágenes. Máximo 5 imágenes de entrada. (Este modelo suele ser el más lento de ejecutar en términos relativos)

> **\* Nota sobre FLUX.2-dev**: Requiere NVIDIA H200.

### Texto a Video y Imagen a Video (Solo LTX-2/LTX-2.3 aceptan T2V y I2V, los demás modelos solo T2V) (`/videos`)

#### Serie Wan2.2
- `Wan-AI/Wan2.2-T2V-A14B` (Alta calidad, 40 pasos - inicia con `--model "wan2.2"`)
- `Aquiles-ai/Wan2.2-Turbo` ⚡ **9.5x más rápido** - ¡La misma calidad en 4 pasos! (inicia con `--model "wan2.2-turbo"`)

#### Serie Wan2.1
- `Wan-AI/Wan2.1-T2V-14B` (Alta calidad, 40 pasos - inicia con `--model "wan2.1"`)
- `Aquiles-ai/Wan2.1-Turbo` ⚡ **9.5x más rápido** - ¡La misma calidad en 4 pasos! (inicia con `--model "wan2.1-turbo"`)
- `Wan-AI/Wan2.1-T2V-1.3B` (Versión ligera, 40 pasos - inicia con `--model "wan2.1-3B"`)
- `Aquiles-ai/Wan2.1-Turbo-fp8` ⚡ **9.5x más rápido + optimizado FP8** - 4 pasos (inicia con `--model "wan2.1-turbo-fp8"`)

#### Serie HunyuanVideo-1.5

**Resolución Estándar (480p)**
- `Aquiles-ai/HunyuanVideo-1.5-480p` (50 pasos - inicia con `--model "hunyuanVideo-1.5-480p"`)
- `Aquiles-ai/HunyuanVideo-1.5-480p-fp8` (50 pasos, optimizado con FP8 - inicia con `--model "hunyuanVideo-1.5-480p-fp8"`)
- `Aquiles-ai/HunyuanVideo-1.5-480p-Turbo` ⚡ **12.5x más rápido** - ¡4 pasos! (inicia con `--model "hunyuanVideo-1.5-480p-turbo"`)
- `Aquiles-ai/HunyuanVideo-1.5-480p-Turbo-fp8` ⚡ **12.5x más rápido + optimizado con FP8** - 4 pasos (inicia con `--model "hunyuanVideo-1.5-480p-turbo-fp8"`)

**Alta Resolución (720p)**
- `Aquiles-ai/HunyuanVideo-1.5-720p` (50 pasos - inicia con `--model "hunyuanVideo-1.5-720p"`)
- `Aquiles-ai/HunyuanVideo-1.5-720p-fp8` (50 pasos, optimizado con FP8 - inicia con `--model "hunyuanVideo-1.5-720p-fp8"`)

#### LTX-2/LTX-2.3 (Generación conjunta de audio y video)

- `Lightricks/LTX-2` (40 steps - start with `--model "ltx-2"`)
- `Lightricks/LTX-2.3` (40 steps - start with `--model "ltx-2.3"`)

> **Características especiales**: LTX-2/LTX-2.3 son los primeros modelos **open-source** que soporta generación sincronizada de audio y video en un único modelo, comparable a modelos cerrados como [Sora-2](https://openai.com/index/sora-2/) y [Veo 3.1](https://gemini.google/cl/overview/video-generation/). Además, LTX-2 soporta **imagen de entrada como primer fotograma** del video - pasa una imagen de referencia mediante `input_reference` para guiar el punto de partida visual de la generación. Para mejores resultados con este modelo, consulta la [guía de prompts](https://ltx.io/model/model-blog/prompting-guide-for-ltx-2) proporcionada por el equipo de Lightricks.

**Ejemplo de imagen a video:**

```bash
curl -X POST "https://TU_BASE_URL_DEPLOY/videos" \
  -H "Authorization: Bearer dummy-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F prompt="She turns around and smiles, then slowly walks out of the frame." \
  -F model="ltx-2" \
  -F size="1280x720" \
  -F seconds="8" \
  -F input_reference="@sample_720p.jpeg;type=image/jpeg"
```

> **Requisitos de VRAM**: La mayoría de los modelos requieren 24GB+ de VRAM. Todos los modelos de video requieren H100/A100-80GB. Las versiones optimizadas con FP8 ofrecen mejor eficiencia de memoria.

[**📖 Documentación completa de modelos**](https://aquiles-ai.github.io/aquiles-image-docs/#models) y más modelos en [**🎬 Aquiles-Studio**](https://huggingface.co/collections/Aquiles-ai/aquiles-studio)

### 🔍 ¿No encuentras el modelo que buscas?

Si el modelo que necesitas no está en la lista nativa, puedes ejecutar prácticamente **cualquier arquitectura** basada en Diffusers (SD 1.5, SDXL, etc.) utilizando nuestra implementación de **AutoPipeline**. 

Consulta la sección de [**🧪 Funcionalidades Avanzadas**](#-funcionalidades-avanzadas) para aprender cómo desplegar cualquier modelo de Hugging Face con un solo comando.

## 💡 Ejemplos

### Generación de Imágenes

https://github.com/user-attachments/assets/00e18988-0472-4171-8716-dc81b53dcafa

https://github.com/user-attachments/assets/00d4235c-e49c-435e-a71a-72c36040a8d7

### Edición de Imágenes

<div align="center">

| Entrada + Prompt | Resultado |
|------------------|-----------|
| <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1764807968/Captura_de_pantalla_1991_as3v28.png" alt="Script de Edición" width="500"/> | <img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1764807952/Captura_de_pantalla_1994_ffmko2.png" alt="Resultado de Edición" width="500"/> |

</div>

### Generación de Videos

https://github.com/user-attachments/assets/7b1270c3-b77b-48df-a0fe-ac39b2320143

> **Nota**: La generación de video con `wan2.2` tarda ~30 minutos en una H100. ¡Con `wan2.2-turbo`, solo tarda ~3 minutos! Solo se puede generar un video a la vez.

**Generación de video y audio**

https://github.com/user-attachments/assets/b7104dc3-5306-4e6a-97e5-93a6c1e73f54

## 🧪 Funcionalidades Avanzadas

### AutoPipeline - Ejecuta Cualquier Modelo de Diffusers

Ejecuta cualquier modelo compatible con `AutoPipelineForText2Image` o `AutoPipelineForImage2Image` desde HuggingFace:

```bash
aquiles-image serve \
  --model "stabilityai/stable-diffusion-xl-base-1.0" \
  --auto-pipeline \
  --set-steps 30 \
  --auto-pipeline-type t2i # o i2i para imagen a imagen
```

**Modelos compatibles incluyen:**
- `stable-diffusion-v1-5/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-xl-base-1.0`
- Cualquier modelo de HuggingFace compatible con `AutoPipelineForText2Image` o `AutoPipelineForImage2Image`

**Consideraciones:**
- ⚠️ Inferencia más lenta que las implementaciones nativas
- ⚠️ Experimental - puede tener problemas de estabilidad

### Soporte de LoRA

Carga cualquier LoRA desde HuggingFace o un path local pasando un archivo JSON de configuración al iniciar el servidor. Compatible con todos los modelos de imagen nativos y AutoPipeline.

**1. Crea un archivo de configuración de LoRA:**

De forma manual:
```json
{
  "repo_id": "brushpenbob/Flux-retro-Disney-v2",
  "weight_name": "Flux_retro_Disney_v2.safetensors",
  "adapter_name": "flux-retro-disney-v2",
  "scale": 1.0
}
```

O de forma programática usando el helper de Python:
```python
from aquilesimage.utils import save_lora_config
from aquilesimage.models import LoRAConfig

save_lora_config(
    LoRAConfig(
        repo_id="brushpenbob/Flux-retro-Disney-v2",
        weight_name="Flux_retro_Disney_v2.safetensors",
        adapter_name="flux-retro-disney-v2"
    ),
    "./lora_config.json"
)
```

**2. Inicia el servidor con LoRA habilitado:**
```bash
aquiles-image serve \
  --model "black-forest-labs/FLUX.1-dev" \
  --load-lora \
  --lora-config "./lora_config.json"
```

Funciona tanto en modo de dispositivo único como en modo distribuido:
```bash
aquiles-image serve \
  --model "black-forest-labs/FLUX.1-dev" \
  --load-lora \
  --lora-config "./lora_config.json" \
  --dist-inference
```

### Modo Dev - Prueba Sin Cargar Modelos

Ideal para desarrollo, pruebas y CI/CD:

```bash
aquiles-image serve --no-load-model
```

**Qué hace:**
- Inicia el servidor al instante sin necesidad de GPU
- Devuelve imágenes de prueba que simulan respuestas reales
- Todos los endpoints funcionan con formatos realistas
- Misma estructura de API que en producción

### Protección con clave de API y Playground

#### Proteger tu servidor con una clave de API

Puedes proteger tu servidor requiriendo una clave de API en cada solicitud. Simplemente pasa `--api-key` al iniciar el servidor:

```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium" --api-key "tu-clave-de-api"
```

Todas las solicitudes deberán incluir la clave en el encabezado `Authorization`:

```bash
curl -X POST "http://localhost:5500/images/generations" \
  -H "Authorization: Bearer tu-clave-de-api" \
  -H "Content-Type: application/json" \
  -d '{"model": "stabilityai/stable-diffusion-3.5-medium", "prompt": "a white siamese cat"}'
```

#### Playground integrado

Aquiles-Image incluye un playground interactivo para probar modelos de imagen y monitorear las estadísticas del servidor — protegido con inicio de sesión para evitar accesos no autorizados. Actívalo con `--username` y `--password`:

```bash
aquiles-image serve --model "stabilityai/stable-diffusion-3.5-medium" \
  --api-key "tu-clave-de-api" \
  --username "root" \
  --password "root"
```

Una vez iniciado, abre `http://localhost:5500` en tu navegador. El playground te permite:
- **Generar imágenes** de forma interactiva usando cualquier modelo de imagen cargado
- **Visualizar estadísticas del servidor** en tiempo real

> **Nota**: El playground solo está disponible para modelos de imagen.

<div align="center">

**Inicio de sesión**

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1772321078/Captura_de_pantalla_2122_pve72l.png" width="700"/>

**Playground**

<img src="https://res.cloudinary.com/dmtomxyvm/image/upload/v1772321078/Captura_de_pantalla_2121_t4k24e.png" width="700"/>

</div>

## 📊 Monitoreo y estadísticas

Aquiles-Image ofrece un endpoint personalizado `/stats` para monitoreo en tiempo real:

```python
import requests

# Obtener estadísticas del servidor
stats = requests.get("http://localhost:5500/stats", 
                    headers={"Authorization": "Bearer TU_CLAVE_DE_API"}).json()

print(f"Total de solicitudes: {stats['total_requests']}")
print(f"Total de imágenes generadas: {stats['total_images']}")
print(f"En cola: {stats['queued']}")
print(f"Completadas: {stats['completed']}")
```

### Formatos de respuesta

La respuesta varía dependiendo del tipo de modelo y la configuración:

#### Modelos de imagen - Modo de dispositivo único

```json
{
  "mode": "single-device",
  "total_requests": 150,
  "total_batches": 42,
  "total_images": 180,
  "queued": 3,
  "completed": 147,
  "failed": 0,
  "processing": true,
  "available": false
}
```

#### Modelos de imagen - Modo distribuido (multi-GPU)

```json
{
  "mode": "distributed",
  "devices": {
    "cuda:0": {
      "id": "cuda:0",
      "available": true,
      "processing": false,
      "can_accept_batch": true,
      "batch_size": 4,
      "max_batch_size": 8,
      "images_processing": 0,
      "images_completed": 45,
      "total_batches_processed": 12,
      "avg_batch_time": 2.5,
      "estimated_load": 0.3,
      "error_count": 0,
      "last_error": null
    },
    "cuda:1": {
      "id": "cuda:1",
      "available": true,
      "processing": true,
      "can_accept_batch": false,
      "batch_size": 2,
      "max_batch_size": 8,
      "images_processing": 2,
      "images_completed": 38,
      "total_batches_processed": 10,
      "avg_batch_time": 2.8,
      "estimated_load": 0.7,
      "error_count": 0,
      "last_error": null
    }
  },
  "global": {
    "total_requests": 150,
    "total_batches": 42,
    "total_images": 180,
    "queued": 3,
    "active_batches": 1,
    "completed": 147,
    "failed": 0,
    "processing": true
  }
}
```

#### Modelos de video

```json
{
  "total_tasks": 25,
  "queued": 2,
  "processing": 1,
  "completed": 20,
  "failed": 2,
  "available": false,
  "max_concurrent": 1
}
```

**Métricas clave:**
- `total_requests/tasks` - Número total de solicitudes de generación recibidas
- `total_images` - Total de imágenes generadas (solo modelos de imagen)
- `queued` - Solicitudes en espera de ser procesadas
- `processing` - Solicitudes procesándose actualmente
- `completed` - Solicitudes completadas con éxito
- `failed` - Solicitudes fallidas
- `available` - Indica si el servidor puede aceptar nuevas solicitudes
- `mode` - Modo de operación para modelos de imagen: `single-device` o `distributed`

## 🎯 Casos de uso

| Quién | Para qué |
|-------|----------|
| 🚀 **Startups de IA** | Construir funciones de generación de imágenes sin costos de API |
| 👨‍💻 **Desarrolladores** | Prototipar con múltiples modelos usando una sola interfaz |
| 🏢 **Empresas** | Infraestructura de IA de imágenes escalable y privada |
| 🔬 **Investigadores** | Experimentar fácilmente con modelos de vanguardia |


## 📋 Requisitos previos

- Python 3.8+
- GPU compatible con CUDA y 24GB+ de VRAM (para la mayoría de los modelos)
- 10GB+ de espacio libre en disco


## 📚 Documentación

- [**Documentación completa**](https://aquiles-ai.github.io/aquiles-image-docs/)
- [**Referencia del cliente**](https://aquiles-ai.github.io/aquiles-image-docs/#client-api)
- [**Guía de modelos**](https://aquiles-ai.github.io/aquiles-image-docs/#models)

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Ya sea que quieras:
- 🐛 Reportar errores y problemas
- 🎨 Agregar soporte para nuevos modelos de imagen
- 📝 Mejorar la documentación

Por favor lee nuestra [**Guía de contribución**](CONTRIBUTING.md) para comenzar.

<div align="center">

**[⭐ Dale una estrella al proyecto](https://github.com/Aquiles-ai/Aquiles-Image)** • **[🐛 Reportar problemas](https://github.com/Aquiles-ai/Aquiles-Image/issues)** • **[🤝 Contribuir](CONTRIBUTING.md)**

*Construido con ❤️ para la comunidad de IA*

</div>