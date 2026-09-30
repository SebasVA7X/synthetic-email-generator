"""
comun.py
Configuración y funciones compartidas entre generar_corpus.py y contar.py.

Todo lo que define "qué es una palabra" está aquí, para que el generador
y el contador apliquen exactamente las mismas reglas. Si se cambia algo
(stopwords, vocabulario), se debe cambiar para los dos scripts a la vez.
"""
import re
import unicodedata

# =====================================================================
# 1. NORMALIZACIÓN
# =====================================================================

# Palabras vacías: artículos, preposiciones, pronombres... No aportan señal
# y ensucian los bigramas ("tus datos", "de la"). Van SIN tildes porque se
# comparan después de normalizar.
# OJO: "no" y "nunca" NO están aquí a propósito. Invierten el sentido
# ("nunca pediremos tus datos") y son lo que el orden 1 debería ver.
STOPWORDS = {
    "a", "al", "ante", "como", "con", "de", "del", "desde", "e", "el", "en",
    "entre", "es", "esa", "ese", "eso", "esta", "estas", "este", "esto",
    "estos", "ha", "han", "has", "hay", "la", "las", "le", "les", "lo", "los",
    "me", "mi", "mis", "nos", "nuestra", "nuestras", "nuestro", "nuestros",
    "o", "para", "pero", "por", "que", "se", "si", "son", "su", "sus", "te",
    "ti", "tu", "tus", "u", "un", "una", "unas", "uno", "unos", "usted",
    "ustedes", "y", "ya",
}


def normalizar(texto, quitar_stopwords=True):
    """Texto -> lista de tokens.

    Pasos: minúsculas, quitar tildes (conservando la ñ), convertir toda la
    puntuación en espacios y, por defecto, quitar palabras vacías.
    Ejemplo: "¡Envía tus DATOS, ya!" -> ["envia", "datos"]
    """
    texto = texto.lower().replace("ñ", "\x00")          # protegemos la ñ
    texto = unicodedata.normalize("NFD", texto)         # "á" -> "a" + tilde suelta
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = texto.replace("\x00", "ñ")                  # devolvemos la ñ
    texto = re.sub(r"[^a-zñ0-9]+", " ", texto)          # puntuación -> espacio
    tokens = texto.split()
    if quitar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def contiene_frase(tokens, frase):
    """¿Aparece la frase (normalizada igual) como secuencia contigua en tokens?"""
    objetivo = normalizar(frase)
    n = len(objetivo)
    return any(tokens[i:i + n] == objetivo for i in range(len(tokens) - n + 1))


# =====================================================================
# 2. VOCABULARIO CONTROLADO
# =====================================================================
# Para cada palabra clave definimos cuántos correos de cada clase la llevarán:
#   p_spam = fracción de correos spam que la contienen    -> P(palabra | Spam)
#   p_ham  = fracción de correos legítimos que la contienen -> P(palabra | Ham)
#
# Son las probabilidades "verdaderas" en el conjunto sintético. El contador
# debería recuperarlas aproximadamente, y esa es una buena prueba de que todo
# funciona. La razón p_spam / p_ham es la razón de verosimilitud de diseño.
#
# "pista" (opcional) ayuda al LLM con palabras que tiende a conjugar o
# cambiar de género (escribe "inmediatamente" en vez de "inmediato").
#
# Las palabras "ambiguas" no se piden sueltas: se piden dentro de una frase
# de contexto que depende de la clase. La palabra sola casi no distingue
# (razón cercana a 1), pero la palabra ANTERIOR sí. Eso es lo que el orden 1
# puede capturar y el orden 0 no.

_VOCAB = [
    # --- Señales fuertes de spam (razón de verosimilitud alta) ---
    {"forma": "felicitaciones", "tipo": "spam", "p_spam": 0.30, "p_ham": 0.04},
    {"forma": "ganador",        "tipo": "spam", "p_spam": 0.35, "p_ham": 0.02,
     "pista": 'en masculino singular, por ejemplo "eres el ganador"'},
    {"forma": "premio",         "tipo": "spam", "p_spam": 0.35, "p_ham": 0.04,
     "pista": "en singular"},
    {"forma": "gratis",         "tipo": "spam", "p_spam": 0.40, "p_ham": 0.06},
    {"forma": "dinero",         "tipo": "spam", "p_spam": 0.30, "p_ham": 0.03},
    {"forma": "urgente",        "tipo": "spam", "p_spam": 0.35, "p_ham": 0.05},
    {"forma": "inmediato",      "tipo": "spam", "p_spam": 0.30, "p_ham": 0.04,
     "pista": 'por ejemplo "de inmediato" o "acceso inmediato"'},
    {"forma": "exclusivo",      "tipo": "spam", "p_spam": 0.25, "p_ham": 0.05,
     "pista": 'en masculino singular, por ejemplo "acceso exclusivo"'},
    {"forma": "regalo",         "tipo": "spam", "p_spam": 0.25, "p_ham": 0.05,
     "pista": "en singular"},

    # --- Señales de correo legítimo (razón baja) ---
    {"forma": "comunidad",      "tipo": "ham",  "p_spam": 0.03, "p_ham": 0.30},
    {"forma": "reunión",        "tipo": "ham",  "p_spam": 0.02, "p_ham": 0.30},
    {"forma": "informe",        "tipo": "ham",  "p_spam": 0.03, "p_ham": 0.30},
    {"forma": "equipo",         "tipo": "ham",  "p_spam": 0.05, "p_ham": 0.30},
    {"forma": "políticas",      "tipo": "ham",  "p_spam": 0.04, "p_ham": 0.25},

    # --- Neutras (razón ≈ 1: aparecen igual en ambas clases) ---
    {"forma": "factura",        "tipo": "neutra", "p_spam": 0.25, "p_ham": 0.25},
    {"forma": "cuenta",         "tipo": "neutra", "p_spam": 0.35, "p_ham": 0.35},
    {"forma": "pago",           "tipo": "neutra", "p_spam": 0.25, "p_ham": 0.25},
    {"forma": "descuento",      "tipo": "neutra", "p_spam": 0.20, "p_ham": 0.20},

    # --- Ambiguas: la palabra sola casi no distingue, el contexto sí ---
    {"forma": "datos", "tipo": "ambigua", "p_spam": 0.30, "p_ham": 0.25,
     "contextos": {"spam": ["envía tus datos", "confirma tus datos", "actualiza tus datos"],
                   "ham":  ["nunca pediremos tus datos", "protegemos tus datos"]}},
    {"forma": "clic", "tipo": "ambigua", "p_spam": 0.30, "p_ham": 0.20,
     "contextos": {"spam": ["haz clic aquí", "da clic en el enlace"],
                   "ham":  ["no hagas clic en enlaces desconocidos",
                            "evita hacer clic en enlaces sospechosos"]}},
    {"forma": "contraseña", "tipo": "ambigua", "p_spam": 0.20, "p_ham": 0.15,
     "contextos": {"spam": ["confirma tu contraseña", "ingresa tu contraseña"],
                   "ham":  ["nunca compartas tu contraseña", "cambia tu contraseña"]}},
]

# Diccionario indexado por la forma normalizada ("reunión" -> "reunion"),
# que es como aparecerá en los tokens.
VOCABULARIO = {normalizar(v["forma"], quitar_stopwords=False)[0]: v for v in _VOCAB}