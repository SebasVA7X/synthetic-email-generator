"""
generar_corpus.py
Genera un corpus sintético de correos (spam / legítimos) con un LLM local vía Ollama.

IDEA CLAVE: se decide qué palabras del vocabulario lleva cada correo
(sorteándolas con las probabilidades de comun.py) y el LLM solo redacta un
correo natural que las contenga. Así las frecuencias quedan bajo control y el
texto no suena a plantilla. Después se valida que el LLM cumplió: si le falta
una palabra o usó una que no debía, le se le pide que corrija.

Uso:
    python generar_corpus.py              # genera todo (entrenamiento + prueba)
    python generar_corpus.py --simular    # prueba el pipeline SIN llamar al LLM
    python generar_corpus.py --max 5      # solo los primeros 5 de cada conjunto

Si se corta a la mitad, vuelve a correrlo: retoma donde quedó y reintenta
los que fallaron. Requiere: pip install requests
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import requests

from comun import VOCABULARIO, contiene_frase, normalizar

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
OLLAMA_URL = "URL MODELO LOCAL"  # sin "/" al final
MODELO = "qwen3.8:27b"      # verifica el nombre exacto del modelo
THINK = False               # Qwen3 razona por defecto; apagarlo acelera mucho.
                            # Pon None si el modelo no acepta el parámetro.
TEMPERATURA = 0.9           # alta = más variedad entre correos
TIMEOUT = 300               # segundos por llamada
MAX_INTENTOS = 5            # intentos por correo antes de darlo por fallido
MAX_PALABRAS = 7            # tope de palabras clave por correo (naturalidad)

N_ENTRENAMIENTO = {"spam": 120, "ham": 80}     # 200 correos, 60/40
N_PRUEBA_ALEATORIA = {"spam": 12, "ham": 8}    # prueba aparte, misma proporción
SEMILLA_ENTRENAMIENTO = 42  # semillas fijas: el plan es reproducible
SEMILLA_PRUEBA = 7          # semilla distinta: la prueba no copia al entrenamiento

ARCHIVO_ENTRENAMIENTO = "corpus_entrenamiento.csv"
ARCHIVO_PRUEBA = "corpus_prueba.csv"

# =====================================================================
# VARIANTES (dan variedad al texto; no afectan las frecuencias)
# =====================================================================
CATEGORIAS = {
    "spam": [
        "un sorteo o lotería falsa que asegura que la persona ganó algo",
        "un intento de phishing que se hace pasar por un banco",
        "una oferta fraudulenta demasiado buena para ser verdad",
        "un cobro falso para que la persona pague o entregue información",
        "un aviso falso de un paquete retenido en aduana",
    ],
    "ham": [
        "una alerta de seguridad real de un banco a su cliente",
        "una confirmación de compra de una tienda en línea",
        "el boletín informativo de una comunidad u organización",
        "un correo de trabajo entre colegas",
        "una promoción legítima de una tienda donde la persona ya es cliente",
    ],
}
ESTILOS = ["formal", "cercano e informal", "breve y directo"]
LARGOS = ["2 a 3 oraciones", "4 a 6 oraciones"]

# =====================================================================
# LOS 3 CASOS DE LA TAREA (van al conjunto de prueba, nunca al entrenamiento)
# =====================================================================
CASOS_TAREA = [
    {   # Spam evidente: debería detectarse con cualquier modelo
        "id": "caso_1", "etiqueta": "spam",
        "categoria": CATEGORIAS["spam"][0], "estilo": "cercano e informal",
        "largo": "4 a 6 oraciones",
        "palabras": ["felicitaciones", "ganador", "premio", "dinero", "inmediato"],
        "frases": ["envía tus datos"],
    },
    {   # Legítimo evidente
        "id": "caso_2", "etiqueta": "ham",
        "categoria": CATEGORIAS["ham"][1], "estilo": "formal",
        "largo": "4 a 6 oraciones",
        "palabras": ["factura", "pago", "comunidad", "politicas", "descuento"],
        "frases": [],
    },
    {   # La trampa: legítimo con palabras "sospechosas". El orden 0 debería
        # equivocarse; el orden 1 debería acertar gracias al contexto.
        "id": "caso_3", "etiqueta": "ham",
        "categoria": CATEGORIAS["ham"][0], "estilo": "formal",
        "largo": "4 a 6 oraciones",
        "palabras": ["urgente", "cuenta"],
        "frases": ["nunca pediremos tus datos",
                   "no hagas clic en enlaces desconocidos",
                   "nunca compartas tu contraseña"],
    },
]

CAMPOS = ["id", "etiqueta", "categoria", "estilo", "asunto", "cuerpo",
          "texto_normalizado", "palabras_objetivo"]

PROMPT_SISTEMA = (
    "Eres un generador de correos electrónicos sintéticos en español para "
    "entrenar un filtro de spam con fines educativos. Sigues las reglas al pie "
    "de la letra y respondes únicamente con JSON."
)

# Ollama fuerza la salida a este esquema JSON (structured outputs)
ESQUEMA = {
    "type": "object",
    "properties": {"asunto": {"type": "string"}, "cuerpo": {"type": "string"}},
    "required": ["asunto", "cuerpo"],
}


# =====================================================================
# PLAN: qué palabras lleva cada correo (se decide ANTES de llamar al LLM)
# =====================================================================
def completar_objetivo(item):
    """Lista de palabras clave que el correo debe contener: las sueltas más
    las ambiguas que vienen dentro de las frases."""
    ambiguas = [t for f in item["frases"] for t in normalizar(f) if t in VOCABULARIO]
    item["objetivo"] = item["palabras"] + ambiguas
    return item


def sortear_palabras(rng, etiqueta):
    """Cada palabra entra con probabilidad p_spam o p_ham según la clase.
    Es literalmente muestrear del modelo que después se intentará recuperar."""
    elegidas = []  # (palabra, frase_de_contexto o None)
    for clave, info in VOCABULARIO.items():
        if rng.random() < info[f"p_{etiqueta}"]:
            frase = rng.choice(info["contextos"][etiqueta]) if info["tipo"] == "ambigua" else None
            elegidas.append((clave, frase))
    if len(elegidas) > MAX_PALABRAS:
        elegidas = rng.sample(elegidas, MAX_PALABRAS)
    palabras = [c for c, f in elegidas if f is None]
    frases = [f for c, f in elegidas if f is not None]
    return palabras, frases


def construir_plan(rng, cantidades, prefijo):
    etiquetas = ["spam"] * cantidades["spam"] + ["ham"] * cantidades["ham"]
    rng.shuffle(etiquetas)
    plan = []
    for i, etiqueta in enumerate(etiquetas, 1):
        palabras, frases = sortear_palabras(rng, etiqueta)
        plan.append(completar_objetivo({
            "id": f"{prefijo}{i:03d}", "etiqueta": etiqueta,
            "categoria": rng.choice(CATEGORIAS[etiqueta]),
            "estilo": rng.choice(ESTILOS), "largo": rng.choice(LARGOS),
            "palabras": palabras, "frases": frases,
        }))
    return plan


# =====================================================================
# PROMPT
# =====================================================================
def formato_palabra(clave):
    """'inmediato (por ejemplo "de inmediato"...)' si la palabra tiene pista."""
    info = VOCABULARIO[clave]
    return f'{info["forma"]} ({info["pista"]})' if "pista" in info else info["forma"]


def construir_prompt(item):
    tipo = ("de spam (fraudulento o no deseado)" if item["etiqueta"] == "spam"
            else "legítimo (real y esperado por quien lo recibe)")
    palabras = [formato_palabra(p) for p in item["palabras"]]
    prohibidas = [v["forma"] for k, v in VOCABULARIO.items() if k not in item["objetivo"]]

    reglas = []
    if palabras:
        reglas.append("Usa EXACTAMENTE estas palabras, tal como están escritas "
                      "(sin cambiar género, número ni conjugación): " + ", ".join(palabras))
    if item["frases"]:
        reglas.append("Incluye LITERALMENTE estas frases, sin cambiar ninguna palabra: "
                      + "; ".join(f'"{f}"' for f in item["frases"]))
    reglas.append("Si una palabra no encaja gramaticalmente, cambia la frase a su "
                  "alrededor, NUNCA la palabra. Por ejemplo, si te piden 'rápido', escribe "
                  "'un envío rápido', no 'rápidamente' ni 'rápida'.")
    reglas.append("NO uses ninguna de estas palabras, ni siquiera en el asunto: "
                  + ", ".join(prohibidas))
    reglas.append("Trata al destinatario de tú en todo el correo (tuteo: confirma, "
                  "envía, revisa), incluso si el estilo es formal. Nunca uses usted.")
    reglas.append("Inventa nombres de empresas y personas; no uses marcas reales.")
    reglas.append("No incluyas enlaces, teléfonos ni direcciones de correo reales.")
    reglas.append("Escribe en español neutro, con naturalidad.")

    reglas_txt = "\n".join(f"{i}. {r}" for i, r in enumerate(reglas, 1))
    return (f"Escribe un correo electrónico {tipo}. Situación: {item['categoria']}.\n"
            f"Estilo: {item['estilo']}. Extensión del cuerpo: {item['largo']}.\n\n"
            f"Reglas obligatorias:\n{reglas_txt}\n\n"
            'Devuelve un JSON con dos campos: "asunto" y "cuerpo".')


# =====================================================================
# LLAMADA AL LLM Y VALIDACIÓN
# =====================================================================
def verificar_ollama():
    """Comprueba conexión y que el modelo exista antes de empezar."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        sys.exit(f"No pude conectar con Ollama en {OLLAMA_URL}: {e}")
    modelos = [m["name"] for m in r.json().get("models", [])]
    if MODELO not in modelos:
        sys.exit(f"El modelo '{MODELO}' no está en el servidor. Disponibles:\n  "
                 + "\n  ".join(modelos))
    print(f"Conectado a {OLLAMA_URL}, modelo {MODELO} OK\n")


def llamar_llm(mensajes):
    payload = {"model": MODELO, "messages": mensajes, "stream": False,
               "format": ESQUEMA, "options": {"temperature": TEMPERATURA}}
    if THINK is not None:
        payload["think"] = THINK
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    datos = json.loads(r.json()["message"]["content"])
    return datos["asunto"].strip(), datos["cuerpo"].strip()


def parecidas(palabra, tokens):
    """Variantes de la palabra que el LLM usó en su lugar
    ("inmediatamente", "inmediata" en vez de "inmediato")."""
    raiz = palabra[:max(4, len(palabra) - 2)]
    return sorted({t for t in tokens if t.startswith(raiz) and t != palabra})


def validar(asunto, cuerpo, item):
    """Devuelve la lista de problemas (vacía = el correo cumple)."""
    tokens = normalizar(asunto + " " + cuerpo)
    presentes = set(tokens)
    problemas = []
    faltan = [p for p in item["palabras"] if p not in presentes]
    if faltan:
        detalles = []
        for p in faltan:
            forma = VOCABULARIO[p]["forma"]
            variantes = parecidas(p, tokens)
            if variantes:  # se le dice QUÉ escribió mal, no solo qué falta
                detalles.append(f"'{forma}' (escribiste {', '.join(variantes)}; "
                                f"debe ser exactamente '{forma}', cambia la frase)")
            else:
                detalles.append(f"'{forma}'")
        problemas.append("faltan estas palabras exactas: " + ", ".join(detalles))
    faltan_frases = [f for f in item["frases"] if not contiene_frase(tokens, f)]
    if faltan_frases:
        detalles = []
        for f in faltan_frases:
            # Se muestra cómo usó la palabra clave (con las 2 palabras previas)
            clave = next(t for t in normalizar(f) if t in VOCABULARIO)
            usos = [" ".join(tokens[max(0, i - 2):i + 1])
                    for i, t in enumerate(tokens) if t == clave]
            detalles.append(f'"{f}"' + (f" (escribiste: {'; '.join(usos)})" if usos else ""))
        problemas.append("faltan estas frases literales: " + "; ".join(detalles))
    sobran = [v["forma"] for k, v in VOCABULARIO.items()
              if k in presentes and k not in item["objetivo"]]
    if sobran:
        problemas.append("usaste palabras prohibidas: " + ", ".join(sobran))
    return problemas


def simular(item):
    """Correo falso para probar el pipeline sin gastar GPU."""
    formas = [VOCABULARIO[p]["forma"] for p in item["palabras"]]
    cuerpo = " ".join(["Texto de relleno."] + [f"Aquí va {w}." for w in formas]
                      + [f.capitalize() + "." for f in item["frases"]])
    return f"Correo simulado {item['id']}", cuerpo


def generar_correo(item, modo_simulado):
    if modo_simulado:
        return simular(item)
    mensajes = [{"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": construir_prompt(item)}]
    for intento in range(1, MAX_INTENTOS + 1):
        try:
            asunto, cuerpo = llamar_llm(mensajes)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"   intento {intento}: error ({e})")
            time.sleep(2)
            continue
        problemas = validar(asunto, cuerpo, item)
        if not problemas:
            if intento > 1:
                print(f"   corregido en el intento {intento}")
            return asunto, cuerpo
        print(f"   intento {intento}: " + " | ".join(problemas))
        # Se le devuelve su propio correo con los problemas para que corrija
        mensajes += [
            {"role": "assistant",
             "content": json.dumps({"asunto": asunto, "cuerpo": cuerpo}, ensure_ascii=False)},
            {"role": "user",
             "content": "Corrige el correo. Problemas: " + "; ".join(problemas)
                        + ". Devuelve el JSON completo corregido."},
        ]
    return None


# =====================================================================
# PROCESAR UN CONJUNTO (con reanudación)
# =====================================================================
def procesar(plan, archivo, modo_simulado):
    existe = os.path.exists(archivo) and os.path.getsize(archivo) > 0
    hechos = set()
    if existe:
        with open(archivo, encoding="utf-8-sig", newline="") as f:
            hechos = {fila["id"] for fila in csv.DictReader(f)}
    pendientes = [it for it in plan if it["id"] not in hechos]
    print(f"== {archivo}: {len(hechos)} ya generados, {len(pendientes)} pendientes ==")

    fallidos = []
    t0 = time.time()
    # utf-8-sig: Excel abre bien las tildes y la ñ
    with open(archivo, "a", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if not existe:
            escritor.writeheader()
        for n, item in enumerate(pendientes, 1):
            print(f"[{n}/{len(pendientes)}] {item['id']} ({item['etiqueta']}): "
                  + ", ".join(item["objetivo"]))
            resultado = generar_correo(item, modo_simulado)
            if resultado is None:
                print("   FALLÓ, se reintentará en la próxima ejecución")
                fallidos.append(item["id"])
                continue
            asunto, cuerpo = resultado
            escritor.writerow({
                "id": item["id"], "etiqueta": item["etiqueta"],
                "categoria": item["categoria"], "estilo": item["estilo"],
                "asunto": asunto, "cuerpo": cuerpo,
                "texto_normalizado": " ".join(normalizar(asunto + " " + cuerpo)),
                "palabras_objetivo": "|".join(item["objetivo"]),
            })
            f.flush()  # se guarda en disco correo por correo
            if not modo_simulado:
                restante = (time.time() - t0) / n * (len(pendientes) - n)
                print(f"   ok (quedan ~{restante / 60:.1f} min)")
    return fallidos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simular", action="store_true",
                        help="no llama al LLM; genera correos de relleno para probar")
    parser.add_argument("--max", type=int, default=None,
                        help="genera solo los primeros N de cada conjunto")
    args = parser.parse_args()

    plan_entrenamiento = construir_plan(random.Random(SEMILLA_ENTRENAMIENTO),
                                        N_ENTRENAMIENTO, "ent_")
    plan_prueba = ([completar_objetivo(dict(c)) for c in CASOS_TAREA]
                   + construir_plan(random.Random(SEMILLA_PRUEBA), N_PRUEBA_ALEATORIA, "pru_"))
    if args.max:
        plan_entrenamiento = plan_entrenamiento[:args.max]
        plan_prueba = plan_prueba[:args.max]

    prefijo = "simulado_" if args.simular else ""
    if not args.simular:
        verificar_ollama()

    fallidos = procesar(plan_entrenamiento, prefijo + ARCHIVO_ENTRENAMIENTO, args.simular)
    fallidos += procesar(plan_prueba, prefijo + ARCHIVO_PRUEBA, args.simular)

    if fallidos:
        print(f"\n{len(fallidos)} correos fallaron: {', '.join(fallidos)}")
        print("Vuelve a correr el script para reintentarlos.")
    else:
        print("\nListo, corpus completo.")


if __name__ == "__main__":
    main()