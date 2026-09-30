"""
contar.py
Lee el corpus, construye las tablas de probabilidad y compara:
  - Markov de orden 0
  - Markov de orden 1

Uso:
    python contar.py
    python contar.py --prefijo simulado_    # para los archivos de --simular

Salidas:
    tabla_orden0.csv        P(palabra | clase) de cada palabra clave
    tabla_transiciones.csv  P(palabra | anterior, clase) de cada transición
    resultados_prueba.csv   clasificación de cada correo de prueba con ambos modelos
    detalle_casos.md        cálculos paso a paso de los 3 casos de la tarea
"""
import argparse
import csv
import math
from collections import Counter

from comun import VOCABULARIO

ALFA = 1.0          # suavizado de Laplace (sumar 1 a todos los conteos)
INICIO = "<inicio>" # "palabra anterior" de la primera palabra del correo
CLASES = ("spam", "ham")


def leer(archivo):
    with open(archivo, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        fila["tokens"] = fila["texto_normalizado"].split()
    return filas


# =====================================================================
# ENTRENAMIENTO = CONTAR
# =====================================================================
def entrenar(filas):
    N = Counter(f["etiqueta"] for f in filas)
    total = sum(N.values())
    prior = {c: N[c] / total for c in CLASES}

    # --- Orden 0: ¿en cuántos correos de cada clase aparece cada palabra? ---

    docs_con = {c: Counter() for c in CLASES}
    for f in filas:
        for palabra in set(f["tokens"]) & VOCABULARIO.keys():
            docs_con[f["etiqueta"]][palabra] += 1

    # Laplace: (conteo + 1) / (N + 2).
    p0 = {c: {p: (docs_con[c][p] + ALFA) / (N[c] + 2 * ALFA) for p in VOCABULARIO}
          for c in CLASES}

    # --- Orden 1: transiciones "anterior -> palabra" en cada clase ---
    bigramas = {c: Counter() for c in CLASES}  # veces que A va seguida de B
    salidas = {c: Counter() for c in CLASES}   # veces que A va seguida de algo
    vocab_total = set()
    for f in filas:
        secuencia = [INICIO] + f["tokens"]
        vocab_total.update(f["tokens"])
        for a, b in zip(secuencia, secuencia[1:]):
            bigramas[f["etiqueta"]][(a, b)] += 1
            salidas[f["etiqueta"]][a] += 1
    V = len(vocab_total) + 1  # +1 reserva probabilidad para palabras nunca vistas

    return {"N": N, "prior": prior, "docs_con": docs_con, "p0": p0,
            "bigramas": bigramas, "salidas": salidas, "V": V}


def p_transicion(m, clase, anterior, palabra):
    """P(palabra | anterior, clase) con Laplace.
    Si 'anterior' nunca se vio en esa clase, da 1/V: sin información.
    Si nunca se vio en NINGUNA clase, da 1/V en ambas y la razón es 1:
    esa transición no aporta evidencia hacia ningún lado."""
    return ((m["bigramas"][clase][(anterior, palabra)] + ALFA)
            / (m["salidas"][clase][anterior] + ALFA * m["V"]))


# =====================================================================
# CLASIFICACIÓN
# =====================================================================
def apariciones(tokens):
    """Primera aparición de cada palabra clave, con la palabra que la precede.
    Un factor por palabra clave en ambos modelos: así la comparación es 1 a 1
    y lo único que cambia entre orden 0 y orden 1 es el contexto."""
    vistos, resultado = set(), []
    secuencia = [INICIO] + tokens
    for anterior, palabra in zip(secuencia, secuencia[1:]):
        if palabra in VOCABULARIO and palabra not in vistos:
            vistos.add(palabra)
            resultado.append((anterior, palabra))
    return resultado


def posterior(logs):
    """Normalizar: de 'puntajes' a probabilidades que suman 1.
    Se trabaja con logaritmos para evitar el underflow."""
    maximo = max(logs.values())
    exps = {c: math.exp(v - maximo) for c, v in logs.items()}
    total = sum(exps.values())
    return {c: exps[c] / total for c in exps}


def clasificar(m, tokens):
    pares = apariciones(tokens)
    # Orden 0: P(clase) × Π P(palabra | clase)
    f0 = [(p, m["p0"]["spam"][p], m["p0"]["ham"][p]) for _, p in pares]
    # Orden 1: P(clase) × Π P(palabra | anterior, clase)
    f1 = [(f"{a} → {p}", p_transicion(m, "spam", a, p), p_transicion(m, "ham", a, p))
          for a, p in pares]
    resultados = {}
    for nombre, factores in (("orden0", f0), ("orden1", f1)):
        logs = {"spam": math.log(m["prior"]["spam"]) + sum(math.log(s) for _, s, _ in factores),
                "ham": math.log(m["prior"]["ham"]) + sum(math.log(h) for _, _, h in factores)}
        resultados[nombre] = {"factores": factores, "logs": logs, "post": posterior(logs)}
    return resultados


# =====================================================================
# SALIDAS
# =====================================================================
def guardar_tabla_orden0(m, archivo):
    filas = []
    for p, info in VOCABULARIO.items():
        ps, ph = m["p0"]["spam"][p], m["p0"]["ham"][p]
        p_spam_dado_w = ps * m["prior"]["spam"] / (ps * m["prior"]["spam"] + ph * m["prior"]["ham"])
        filas.append({
            "palabra": p, "tipo_diseno": info["tipo"],
            "n_spam": m["docs_con"]["spam"][p], "n_ham": m["docs_con"]["ham"][p],
            "P(w|Spam)": round(ps, 4), "P(w|Ham)": round(ph, 4),
            "razon_verosimilitud": round(ps / ph, 3),
            "P(Spam|w)": round(p_spam_dado_w, 4),
            # Compara estas dos con las de arriba: ¿recuperamos el diseño?
            "diseno_p_spam": info["p_spam"], "diseno_p_ham": info["p_ham"],
        })
    filas.sort(key=lambda f: -f["razon_verosimilitud"])
    escribir_csv(archivo, filas)
    return filas


def guardar_tabla_transiciones(m, archivo):
    pares = {(a, b) for c in CLASES for (a, b) in m["bigramas"][c] if b in VOCABULARIO}
    filas = []
    for a, b in pares:
        ps, ph = p_transicion(m, "spam", a, b), p_transicion(m, "ham", a, b)
        filas.append({
            "anterior": a, "palabra": b,
            "n_spam": m["bigramas"]["spam"][(a, b)], "n_ham": m["bigramas"]["ham"][(a, b)],
            "P(w|ant,Spam)": f"{ps:.5f}", "P(w|ant,Ham)": f"{ph:.5f}",
            "razon_verosimilitud": round(ps / ph, 3),
        })
    filas.sort(key=lambda f: (f["palabra"], -(f["n_spam"] + f["n_ham"])))
    escribir_csv(archivo, filas)


def escribir_csv(archivo, filas):
    with open(archivo, "w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def detalle_markdown(m, fila, res):
    real = fila["etiqueta"]
    lineas = [f"## {fila['id']} (etiqueta real: {real})", "",
              f"**Asunto:** {fila['asunto']}", "", f"**Cuerpo:** {fila['cuerpo']}", "",
              f"**Texto normalizado:** `{fila['texto_normalizado']}`", ""]
    titulos = {"orden0": ("Orden 0 (Naive Bayes)", "Palabra", "P(w \\| Spam)", "P(w \\| Ham)"),
               "orden1": ("Orden 1 (transiciones)", "Transición",
                          "P(w \\| anterior, Spam)", "P(w \\| anterior, Ham)")}
    for nombre, (titulo, col, cs, ch) in titulos.items():
        r = res[nombre]
        lineas += [f"### {titulo}", "", f"| {col} | {cs} | {ch} | Razón |", "|---|---|---|---|"]
        for etiqueta, s, h in r["factores"]:
            lineas.append(f"| {etiqueta} | {s:.5f} | {h:.5f} | {s / h:.2f} |")
        prod_s, prod_h = math.exp(r["logs"]["spam"]), math.exp(r["logs"]["ham"])
        p_spam = r["post"]["spam"]
        pred = "spam" if p_spam >= 0.5 else "ham"
        lineas += ["",
                   f"- Prior: P(Spam) = {m['prior']['spam']:.3f}, P(Ham) = {m['prior']['ham']:.3f}",
                   f"- P(Spam) × Π factores = {prod_s:.4e}",
                   f"- P(Ham) × Π factores = {prod_h:.4e}",
                   f"- **P(Spam | correo) = {prod_s:.4e} / ({prod_s:.4e} + {prod_h:.4e}) = {p_spam:.4f}**",
                   f"- Clasificación: **{pred}** {'✔' if pred == real else '✘'}", ""]
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefijo", default="", help='p. ej. "simulado_"')
    args = parser.parse_args()

    entrenamiento = leer(args.prefijo + "corpus_entrenamiento.csv")
    prueba = leer(args.prefijo + "corpus_prueba.csv")
    m = entrenar(entrenamiento)

    print(f"Entrenamiento: {m['N']['spam']} spam, {m['N']['ham']} ham "
          f"(prior P(Spam) = {m['prior']['spam']:.2f}), vocabulario V = {m['V']}\n")

    tabla = guardar_tabla_orden0(m, "tabla_orden0.csv")
    print(f"{'palabra':<16}{'n_spam':>7}{'n_ham':>7}{'P(w|S)':>9}{'P(w|H)':>9}"
          f"{'razón':>8}   diseño")
    for f in tabla:
        print(f"{f['palabra']:<16}{f['n_spam']:>7}{f['n_ham']:>7}{f['P(w|Spam)']:>9.3f}"
              f"{f['P(w|Ham)']:>9.3f}{f['razon_verosimilitud']:>8.2f}   "
              f"{f['diseno_p_spam']:.2f} / {f['diseno_p_ham']:.2f}")
    guardar_tabla_transiciones(m, "tabla_transiciones.csv")

    # --- Evaluar en el conjunto de prueba ---
    resultados, detalle = [], ["# Detalle de los casos de la tarea", ""]
    aciertos = {"orden0": 0, "orden1": 0}
    errores = {"orden0": Counter(), "orden1": Counter()}
    for fila in prueba:
        res = clasificar(m, fila["tokens"])
        salida = {"id": fila["id"], "etiqueta_real": fila["etiqueta"],
                  "n_palabras_clave": len(res["orden0"]["factores"])}
        for nombre in ("orden0", "orden1"):
            p = res[nombre]["post"]["spam"]
            pred = "spam" if p >= 0.5 else "ham"
            salida[f"{nombre}_P_spam"] = round(p, 4)
            salida[f"{nombre}_prediccion"] = pred
            if pred == fila["etiqueta"]:
                aciertos[nombre] += 1
            else:
                errores[nombre]["falso positivo" if pred == "spam" else "falso negativo"] += 1
        resultados.append(salida)
        if fila["id"].startswith("caso_"):
            detalle.append(detalle_markdown(m, fila, res))

    escribir_csv("resultados_prueba.csv", resultados)
    with open("detalle_casos.md", "w", encoding="utf-8") as f:
        f.write("\n".join(detalle))

    n = len(prueba)
    print(f"\nPrueba ({n} correos):")
    for nombre in ("orden0", "orden1"):
        e = errores[nombre]
        print(f"  {nombre}: exactitud {aciertos[nombre] / n:.1%}  "
              f"(falsos positivos: {e['falso positivo']}, falsos negativos: {e['falso negativo']})")
    print("\nCasos de la tarea:")
    for r in resultados:
        if r["id"].startswith("caso_"):
            print(f"  {r['id']} (real {r['etiqueta_real']}): "
                  f"orden 0 -> {r['orden0_prediccion']} ({r['orden0_P_spam']:.3f}), "
                  f"orden 1 -> {r['orden1_prediccion']} ({r['orden1_P_spam']:.3f})")
    print("\nArchivos: tabla_orden0.csv, tabla_transiciones.csv, "
          "resultados_prueba.csv, detalle_casos.md")


if __name__ == "__main__":
    main()
