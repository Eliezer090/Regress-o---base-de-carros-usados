import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

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
     
"""Pré-processamento"""
"""
inconsistentes
loc = Realizar localização de determinados regitros com determinadas caracteristicas
"""
i1 = base.loc[base.price <= 10]
#Média do preço
#base.price.mean()
#salva dentro da variavel base os veiculos aonde o preço é maior que 10
base = base[base.price>10]
i2=base.loc[base.price>350000]
base = base[base.price<350000]

"""VALORES FALTANTES"""
#Localiza valores null de uma coluna
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell

base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein

#Preencher os valores nulos com os mais repetidos na base
valores = {'vehicleType':'limousine','gearbox':'manuell','model':'golf','fuelType':'benzin','notRepairedDamage':'nein'}
#realizar a substituição dos valores nullos para os informados acima.
base = base.fillna(value = valores)

"""Prepara as variaveis que iremos utilizar na rede neural """
"""
Os : quer dizer que vai pegar todas as linhas
e o 1:13 pega todas as colunas do 1 até o 13 nao pega a 0 pois é o que queremos prever
"""
previsores = base.iloc[:, 1:13].values
#Aqui extrai somnete os preços
preco_real = base.iloc[:, 0].values

"""
    Label Encoder
É uma função para transformar os dados categóricos em numericos, pois temos dados
strings e os algoritimos nao intendem isso
"""
labelencoder_previsores = LabelEncoder()
#Transformando os valores do abtest em numeric
previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder_previsores.fit_transform(previsores[:,10])
"""Fim label encoder"""

"""
Para os valores transformados pelo Label Encoder teremos que fazer o one hot encoder que seria separar eles da seguinte maneira:
    Se tivessemos somente 0 1 2 teriamos:
Para o 0: 0 0 0
Para o 1: 0 1 0
Para o 2: 0 0 1

Realizando o one hot encoder
categorical_features = passamos os mesmos indices do de cima para ele realizar o encoder
"""
#onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,8,9,10]) LINHA DE BAIXO SUBSTITUI ESSA ATUALIZAÇÃO
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()


"""FIM DO AJUSTANDO A BASE"""

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
regressor.fit(previsores,preco_real,batch_size=300,epochs=200)    
    
previsoes = regressor.predict(previsores)

#media de preços
preco_real.mean()
previsores.mean()