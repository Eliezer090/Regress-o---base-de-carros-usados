import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('/Users/es19237/Desktop/Deep Learning/regressa um valor/files/autos.csv', encoding='ISO-8859-1')
"""AJUSTANDO A BASE"""
#Dropando as colunas que nao vamos precisar
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
"""Analisadno registros que tem uma variabilidade muito grande que nao seria possivel achar uma coorelação"""
#Agrupa e conta registros iguais
base['name'].value_counts()
#Sim apagamos o name ele nao faz tanta falta pois temos a marca
base = base.drop('name', axis=1)
base['seller'].value_counts()
base = base.drop('seller', axis=1)
base['offerType'].value_counts()
base = base.drop('offerType', axis=1)

base = base[base.price>10]
i2=base.loc[base.price>350000]
base = base[base.price<350000]

valores = {'vehicleType':'limousine','gearbox':'manuell','model':'golf','fuelType':'benzin','notRepairedDamage':'nein'}
#realizar a substituição dos valores nullos para os informados acima.
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
#Aqui extrai somnete os preços
preco_real = base.iloc[:, 0].values

labelencoder_previsores = LabelEncoder()
#Transformando os valores do abtest em numeric
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


def criarRede():
    """Iniciando a criação da rede neural"""
    regressor = Sequential()
    """
    Mesmo para essa previsao de valores(Regressão) a relu é a melhor ainda
    Para achar o units: 316(qtde de colunas)+1/2
    input_dim é a qtde de clunas
    """
    regressor.add(Dense(units = 159, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 159, activation = 'relu'))
    #Linear é para nenhuma pois aqui nao precisamos pois queremos o valor que a rede achou mesmo
    regressor.add(Dense(units = 1, activation='linear'))
    """
    mean_absolute_error = media sem considerar o sinal que o calculo deu, ex: 100-200 = -100, com a função absolute ele tirra o negativo
    """
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return regressor


regressor = KerasRegressor(build_fn=criarRede,epochs=100,batch_size=300)
resultados = cross_val_score(estimator=regressor, X=previsores, y=preco_real, cv=10, scoring='neg_mean_absolute_error')

media = resultados.mean()
desvio = resultados.std()












