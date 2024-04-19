# TFG

En este repositorio se guardará todo el codigo utilizado y explicado para la realización de mi TFG: Clasificación de kaones/piones a través de modelos de machine learning y deep learning.

Comentaremos los distintos notebook y lo que que contiene cada uno:

### read_datas.ipynb

Aquí podemos encontrar toda la información necesaria para leer los datos, almacenarlos en un dataframe de pandas. Se explican cada una de las variables y sus distintos valores posibles. Vemos si hay casos donde no contienen información las distintas líneas del dataframe. Se realiza transformaciones de los datos y se separa el conjunto en *train*, *valid* y *test* para realizar las pruebas oportunas. Estos datos se escriben en *.csv* para poder ser leidos por los siguientes notebook y no almacenarlos en la memoria RAM.


### estadisticas.ipynb

Cargamos los datos de train y mostramos las distintas relaciones que hay entre cada una de las variables. Mostramos la distribución de las distintas varaibles para conocer su comportamiento.Se comenta que las features que usaremos para el entrenamiento serán *['hitX','hitY', 'hitZ', 'hitInteg']*. También se puede ver una distribución espacial de 300 kaones y 300 piones para ver como se comportan. Para poder entrenar los modelos de Machine Learning necesitamos una estructura fija, para ello necesitamos definir un N que será el número de hits que serán representados por eventos. Para ello se define la funcion **filtrado_datos**, la que nos devolverá una matriz donde cada fila será un eventos y un array de etiquetas donde contendrá la etiqueta de la partícula, 0 para los piones y 1 para los kaones. Si el N es mayor que el número de hits por evento se completa con ceros, si es menor, se ordenan por orden cronológico y nos quedamos con las mayores. Se hace un estudio del numero de hits por eventos y nos quedamos con los valores mas representatativos. Estos estan en el intervalo [400,800]. También se vé que no hay un desbalanceo de clases.


### modelosML.ipynb

Entrenaremos varios modelos de Machine Learning, desde los más complejos a los más sencillo para ver la diferencia que hay entre ellos. También se entrenará con las distintas N para ver con que N trabaja mejor cada modelo. Para ello se entrenará en el intervalo N=[400,800] en saltos de 40. Al final mostraremos una comparativa entre las métricas de los distintos modelos.


### CVML.ipynb

Unimos los datos de validacion y train y entrenamos los 4 mejores modelos por CV para ver cual es el que mejor resultados da. Al final se muestra una comparativa entre los 4 modelos.

### explicabilidad.ipynb

Cojeremos el mejor modelo en cuanto a resultados y tiempo de entrenamiento del anterior notebook y veremos su comportamiento en la clasificación en bajas-medias-altas energías. Para clasificar estas zonas se cogen todos los valores de las energías, se ordenan y se dividen entre 3 para ver cual es el corte. Finalemte se entrena todo el modelo con los datos de validación y train y se obtienen los datos de accuracy en test.

Se guarda el modelo para poder usarlo en la regresión.


### entrenamiento_regresion.ipynb


