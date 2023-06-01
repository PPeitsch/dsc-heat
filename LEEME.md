# Determinación de Temperatura y Entalpía de transformaciones en curvas de DSCx

**ESTRUCTURA DE ARCHIVO:**

Tipo de datos aceptados: **ASCII**, codificación: **ANSI**
- Las columnas a leer son 5:
    1. Indice (ºC)
    2. Tiempo (s)
    3. Temperatura del horno (ºC)
    4. Temperatura de la muestra (ºC)
    5. Termogravimetría (mg)
    6. Flujo de calor (mW)

**ESTRUCTURA DE NOMBRE:**
* Nombre de archivo de curva a procesar: "nombre.txt" # Admite espacios
* Nombre de archivo de curva instrumental: "Blanco nombre.txt" # Admite espacios

**EJEMPLO:**
* Curva a procesar: "Cp Aluminio.txt"
* Curva instrumental: "Blanco Cp Aluminio.txt"

**DETALLES Y RECOMENDACIONES:**
1. Se recomienda utilizar mismo tiempo de adquisición en curva original e instrumental
2. Se pueden cambiar las rutas de acceso donde se leerán los archivos a procesar (experimental)
3. Asegurarse de solo colocar archivos a procesar en las carpetas correspondientes por precausión
4. Las carpetas deben existir