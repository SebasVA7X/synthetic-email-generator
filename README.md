# Clasificación de correos mediante modelos de Markov

Este repositorio contiene los scripts utilizados para generar y analizar el corpus sintético empleado en el ejercicio de clasificación de correos electrónicos como **spam** o **legítimos (ham)**.

El experimento compara dos aproximaciones:

- **Orden 0:** utiliza la probabilidad de aparición de cada palabra según la clase, sin considerar su posición.
- **Orden 1:** incorpora la palabra anterior para evaluar el contexto mediante transiciones entre palabras.

## Estructura

```text
.
├── comun.py
├── generar_corpus.py
├── contar.py
├── corpus_entrenamiento.csv
├── corpus_prueba.csv
├── detalle_casos.md
├── resultados_prueba.csv
├── tabla_orden0.csv
├── tabla_transiciones.csv
└── README.md
```

### `comun.py`

Contiene la configuración compartida por los demás scripts:

- vocabulario controlado de palabras asociadas a spam, correos legítimos, palabras neutras y ambiguas;
- probabilidades esperadas de aparición de cada palabra según la clase;
- normalización del texto;
- eliminación de palabras vacías (*stopwords*);
- tratamiento de palabras ambiguas cuyo significado depende del contexto.

Centralizar estas reglas permite que la generación y el análisis del corpus utilicen exactamente el mismo procedimiento de normalización.

### `generar_corpus.py`

Genera el corpus sintético utilizado en el experimento.

Primero determina, mediante probabilidades previamente definidas, qué palabras debe contener cada correo. Posteriormente utiliza un modelo de lenguaje local mediante **Ollama** para redactar un correo natural que respete esas condiciones.

El script valida automáticamente el resultado y solicita una corrección cuando el modelo:

- omite una palabra requerida;
- modifica una palabra que debía aparecer literalmente;
- no respeta una frase de contexto;
- introduce una palabra del vocabulario que no debía aparecer.

El conjunto de entrenamiento contiene **200 correos**, distribuidos en:

- 120 spam;
- 80 legítimos.

También se genera un conjunto independiente de prueba que incluye los tres casos utilizados en el ejercicio.

Las semillas aleatorias están fijadas para permitir reproducir la selección de palabras del experimento.

### `contar.py`

Procesa los corpus generados y calcula las probabilidades utilizadas para la clasificación.

Para **orden 0**, estima:

```text
P(palabra | Spam)
P(palabra | Ham)
```

mediante la frecuencia de aparición de cada palabra en los documentos de cada clase y suavizado de Laplace.

Para **orden 1**, calcula probabilidades de transición:

```text
P(palabra | palabra anterior, clase)
```

Esto permite considerar el contexto inmediato de una palabra. Por ejemplo, una palabra potencialmente ambigua puede aportar evidencia diferente dependiendo de la palabra que la preceda.

Finalmente, el script clasifica los correos del conjunto de prueba y compara los resultados obtenidos por ambos modelos.

## Flujo del experimento

```text
comun.py
   │
   ├── vocabulario y probabilidades
   └── reglas de normalización
          │
          ▼
generar_corpus.py
          │
          ├── selecciona palabras según probabilidades
          ├── genera los correos mediante Ollama
          ├── valida el contenido
          └── normaliza el texto
          │
          ▼
 corpus_entrenamiento.csv
 corpus_prueba.csv
          │
          ▼
      contar.py
          │
          ├── estima probabilidades de orden 0
          ├── estima transiciones de orden 1
          ├── clasifica los correos de prueba
          └── genera tablas y resultados
```

## Ejecución

### Requisitos

- Python 3
- `requests`
- Ollama, únicamente para generar el corpus mediante el modelo local configurado en `generar_corpus.py`

Instalar la dependencia de Python:

```bash
pip install requests
```

### 1. Generar el corpus

```bash
python generar_corpus.py
```

Para probar el proceso sin realizar llamadas al modelo local:

```bash
python generar_corpus.py --simular
```

También puede limitarse la cantidad de elementos generados:

```bash
python generar_corpus.py --max 5
```

La ejecución produce principalmente:

```text
corpus_entrenamiento.csv
corpus_prueba.csv
```

### 2. Analizar el corpus

Una vez generados los archivos:

```bash
python contar.py
```

El análisis produce:

```text
tabla_orden0.csv
tabla_transiciones.csv
resultados_prueba.csv
detalle_casos.md
```

`tabla_orden0.csv` contiene las probabilidades estimadas para cada palabra y clase.

`tabla_transiciones.csv` contiene las probabilidades correspondientes a las transiciones utilizadas en el modelo de orden 1.

`resultados_prueba.csv` contiene la clasificación obtenida para cada correo mediante ambos modelos.

`detalle_casos.md` presenta los cálculos detallados correspondientes a los tres casos utilizados en el ejercicio.

## Propósito

El código fue desarrollado como apoyo reproducible para el ejercicio académico. Su objetivo es permitir observar cómo cambia la clasificación cuando se consideran únicamente las palabras presentes en un correo frente a un modelo que incorpora también su contexto inmediato.

## Nota sobre uso de IA

El código fue escrito y documentado con el apoyo de Claude Code; los cálculos específicos presentados en la tarea y su anexo se comprobaron manualmente para descartar alucinaciones y otros potenciales errores. 
