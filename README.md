# SIA-TP3

## Instrucciones de configuración y ejecución
Para usar las librerías requeridas, utilizar los siguientes comandos (dependiendo de la versión de Python que se quiera utilizar):
```javascript
pip install numpy | pip3 install numpy
pip install matplotlib | pip3 install matplotlib
```

## Ejercicio 1
Para configurar el ejercicio 1, debemos especificar en el archivo ```input.json``` que el tipo de perceptrón es simple, y debemos aclarar qué tipo de función lógica deseamos resolver. Ejemplo:

```javascript
    {
        "PERCEPTRON":"SIMPLE",
        "FUNCTION":"AND",
        "LEARNING_RATE":0.01,
        "ADAPTIVE_LEARNING_RATE":"TRUE",
        "BETA": 0.5,
        "EPOCHS":100,
        "ERROR_TOLERANCE":0,
        "CLASSIFICATION_MARGIN":0.15
    }
```

Donde el campo ```"FUNCTION"```puede tomar los valores ```"AND"```o ```"XOR"```.

## Ejercicio 2
Para configurar el ejercicio 2, debemos especificar en el archivo ```input.json``` que el tipo de perceptrón es simple, y debemos aclarar que el ejercicio que deseamos resolver es el 2. Ejemplo:

```javascript
    {
        "PERCEPTRON":"SIMPLE",
        "FUNCTION":"Ej2",
        "LEARNING_RATE":0.01,
        "ADAPTIVE_LEARNING_RATE":"TRUE",
        "BETA": 0.5,
        "EPOCHS":100,
        "ERROR_TOLERANCE":0,
        "CLASSIFICATION_MARGIN":0.15
    }
```

## Ejercicio 3
Para configurar el ejercicio 3, debemos especificar en el archivo ```input.json``` que el tipo de perceptrón es multicapa, y debemos aclarar qué problema deseamos resolver. Ejemplo:

```javascript
    {
        "PERCEPTRON":"MULTI",
        "FUNCTION":"XOR",
        "LEARNING_RATE":0.01,
        "ADAPTIVE_LEARNING_RATE":"TRUE",
        "BETA": 0.5,
        "EPOCHS":100,
        "ERROR_TOLERANCE":0,
        "CLASSIFICATION_MARGIN":0.15
    }
```

Donde el campo ```"FUNCTION"```puede tomar los valores ```"EVEN"```o ```"XOR"```.

Luego para ejecutar el programa, dentro de la carpeta ```src```, corremos el siguiente comando:

```javascript
python main.py | python3 main.py
```

## Aclaraciones sobre archivo ```input.json```

Opciones posibles para los campos del archivo de configuración

### Cómo utilizarlo

```javascript
{
  "CAMPO":"VALOR",
  "CAMPO":VALOR_NUMERICO
}
```

|            CAMPO           |                                VALOR                                |                                                                                               DETALLE                                                                                               |
|:--------------------------:|:-------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          PERCEPTRON         |                                SIMPLE                               |                                                                                                                                                                                                     |
|                            |                                 MULTI                                 |                                                                                                                                                                                                     |
|          FUNCTION         |                             AND                            |        Intenta resolver una separación lineal a partir del problema de la función lógica AND                                                                                                    |
|                            |                              XOR                              |    Intenta resolver una separación lineal si se ejecuta con perceptrón simple a partir del problema de la función lineal XOR o intenta aprender el mismo problema a partir de un perceptrón multicapa                                                      |
|                            |                               Ej2                               |       Ejecuta dos instancias de perceptrones, uno simple lineal y otro simple no lineal, y entrena a partir de un conjunto de entrenamiento para luego realizar una corrida de prueba con las redes ya entrenadas                                         |
|                            |                               EVEN                               |        Intenta resolver una categorización de números pares e impares a partir de imagenes de píxeles que representan los números del 0 al 9                                                                  |
|          LEARNING_RATE          |                                 VALOR NUMÉRICO                                | Determina la tasa de aprendizaje del perceptrón                                                                    |
|       ADAPTIVE_LEARNING_RATE       |                             TRUE / FALSE                             | Determina si el perceptrón adapta su tasa de aprendizaje en base al error por épocas |
| BETA | VALOR NUMÉRICO                                | Determina el término beta presente en la función logística o tanh de los perceptrones simples no lineales o multicapa |
| EPOCHS     |                             VALOR NUMÉRICO                             | Determina la máxima cantidad de épocas con las que se entrenará al perceptrón                                      |
|ERROR_TOLERANCE            |                                 VALOR NUMÉRICO                                | |
|           CLASSIFICATION_MARGIN           |                                 VALOR_NUMÉRICO                                 |                                                                                                                   |