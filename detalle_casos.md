# Detalle de los casos de la tarea

## caso_1 (etiqueta real: spam)

**Asunto:** Felicitaciones por tu suerte

**Cuerpo:** Hola, quería decirte que eres el ganador del sorteo de la empresa Fantasía S.L. Te aseguro que recibirás el premio de dinero de inmediato. Solo tienes que envía tus datos para completar el proceso. No pierdas la oportunidad, actúa ahora mismo. Saludos cordiales.

**Texto normalizado:** `felicitaciones suerte hola queria decirte eres ganador sorteo empresa fantasia s l aseguro recibiras premio dinero inmediato solo tienes envia datos completar proceso no pierdas oportunidad actua ahora mismo saludos cordiales`

### Orden 0 (Naive Bayes)

| Palabra | P(w \| Spam) | P(w \| Ham) | Razón |
|---|---|---|---|
| felicitaciones | 0.23770 | 0.01220 | 19.49 |
| ganador | 0.41803 | 0.02439 | 17.14 |
| premio | 0.28689 | 0.02439 | 11.76 |
| dinero | 0.28689 | 0.02439 | 11.76 |
| inmediato | 0.31148 | 0.04878 | 6.39 |
| datos | 0.37705 | 0.23171 | 1.63 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 5.7629e-04
- P(Ham) × Π factores = 7.9998e-10
- **P(Spam | correo) = 5.7629e-04 / (5.7629e-04 + 7.9998e-10) = 1.0000**
- Clasificación: **spam** ✔

### Orden 1 (transiciones)

| Transición | P(w \| anterior, Spam) | P(w \| anterior, Ham) | Razón |
|---|---|---|---|
| <inicio> → felicitaciones | 0.00787 | 0.00081 | 9.69 |
| eres → ganador | 0.04170 | 0.00174 | 24.02 |
| recibiras → premio | 0.00260 | 0.00087 | 2.99 |
| premio → dinero | 0.00252 | 0.00087 | 2.90 |
| dinero → inmediato | 0.00084 | 0.00087 | 0.97 |
| envia → datos | 0.01355 | 0.00087 | 15.65 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 1.4724e-14
- P(Ham) × Π factores = 3.2029e-19
- **P(Spam | correo) = 1.4724e-14 / (1.4724e-14 + 3.2029e-19) = 1.0000**
- Clasificación: **spam** ✔

## caso_2 (etiqueta real: ham)

**Asunto:** Confirmación de tu pedido en Solara Textil

**Cuerpo:** Te confirmamos que tu pedido ha sido procesado correctamente y el pago se ha registrado sin incidencias. Te enviamos la factura detallada para que la revises y la guardes en tu archivo. Si tienes cualquier duda sobre las políticas de devolución, puedes consultarlas en la sección de ayuda. Recuerda que tu comunidad de clientes accede a un descuento adicional en la próxima temporada. Agradecemos tu confianza y esperamos que disfrutes tu nueva prenda.

**Texto normalizado:** `confirmacion pedido solara textil confirmamos pedido sido procesado correctamente pago registrado sin incidencias enviamos factura detallada revises guardes archivo tienes cualquier duda sobre politicas devolucion puedes consultarlas seccion ayuda recuerda comunidad clientes accede descuento adicional proxima temporada agradecemos confianza esperamos disfrutes nueva prenda`

### Orden 0 (Naive Bayes)

| Palabra | P(w \| Spam) | P(w \| Ham) | Razón |
|---|---|---|---|
| pago | 0.22131 | 0.31707 | 0.70 |
| factura | 0.22131 | 0.30488 | 0.73 |
| politicas | 0.03279 | 0.18293 | 0.18 |
| comunidad | 0.04098 | 0.32927 | 0.12 |
| descuento | 0.16393 | 0.31707 | 0.52 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 6.4735e-06
- P(Ham) × Π factores = 7.3847e-04
- **P(Spam | correo) = 6.4735e-06 / (6.4735e-06 + 7.3847e-04) = 0.0087**
- Clasificación: **ham** ✔

### Orden 1 (transiciones)

| Transición | P(w \| anterior, Spam) | P(w \| anterior, Ham) | Razón |
|---|---|---|---|
| correctamente → pago | 0.00087 | 0.00086 | 1.01 |
| enviamos → factura | 0.00087 | 0.00173 | 0.50 |
| sobre → politicas | 0.00172 | 0.00173 | 0.99 |
| recuerda → comunidad | 0.00087 | 0.00170 | 0.51 |
| accede → descuento | 0.00087 | 0.00087 | 1.00 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 5.8637e-16
- P(Ham) × Π factores = 1.5264e-15
- **P(Spam | correo) = 5.8637e-16 / (5.8637e-16 + 1.5264e-15) = 0.2775**
- Clasificación: **ham** ✔

## caso_3 (etiqueta real: ham)

**Asunto:** Alerta de seguridad: actividad inusual detectada

**Cuerpo:** Te contactamos para informarte sobre una actividad urgente relacionada con tu cuenta. Recuerda que nunca pediremos tus datos por este medio. Por tu seguridad, no hagas clic en enlaces desconocidos. También debes tener presente que nunca compartas tu contraseña con nadie. Confirma tus operaciones directamente desde la app oficial de Bancos Virtuales.

**Texto normalizado:** `alerta seguridad actividad inusual detectada contactamos informarte sobre actividad urgente relacionada cuenta recuerda nunca pediremos datos medio seguridad no hagas clic enlaces desconocidos tambien debes tener presente nunca compartas contraseña nadie confirma operaciones directamente app oficial bancos virtuales`

### Orden 0 (Naive Bayes)

| Palabra | P(w \| Spam) | P(w \| Ham) | Razón |
|---|---|---|---|
| urgente | 0.27869 | 0.01220 | 22.85 |
| cuenta | 0.37705 | 0.36585 | 1.03 |
| datos | 0.37705 | 0.23171 | 1.63 |
| clic | 0.29508 | 0.20732 | 1.42 |
| contraseña | 0.14754 | 0.17073 | 0.86 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 1.0350e-03
- P(Ham) × Π factores = 1.4637e-05
- **P(Spam | correo) = 1.0350e-03 / (1.0350e-03 + 1.4637e-05) = 0.9861**
- Clasificación: **spam** ✘

### Orden 1 (transiciones)

| Transición | P(w \| anterior, Spam) | P(w \| anterior, Ham) | Razón |
|---|---|---|---|
| actividad → urgente | 0.00087 | 0.00086 | 1.01 |
| relacionada → cuenta | 0.00087 | 0.00087 | 1.00 |
| pediremos → datos | 0.00087 | 0.00691 | 0.13 |
| hagas → clic | 0.00087 | 0.00947 | 0.09 |
| compartas → contraseña | 0.00087 | 0.00863 | 0.10 |

- Prior: P(Spam) = 0.600, P(Ham) = 0.400
- P(Spam) × Π factores = 2.9701e-16
- P(Ham) × Π factores = 1.6920e-13
- **P(Spam | correo) = 2.9701e-16 / (2.9701e-16 + 1.6920e-13) = 0.0018**
- Clasificación: **ham** ✔
