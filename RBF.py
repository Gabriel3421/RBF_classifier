'''
Aluno: Gabriel de Souza Nogueira da Silva
Matricula: 398847
'''
import re
import numpy as np
import math
from scipy import stats
from sklearn.cluster import KMeans


num_neuronio_oculto = 10

centroides = []
            
cont = 0
x1 = []
x2 = []
y = []

mat_att_treino = np.ones((800,2))
mat_resp_treino = np.ones((800,1))
mat_att_teste = np.ones((201,2))
mat_resp_teste = np.ones((201,1))


dados = open("twomoons.dat", "r")
for line in dados:
    # separando o que é x do que é d
    line = line.strip()  # quebra no \n
    line = re.sub('\s+', ',', line)  # trocando os espaços vazios por virgula
    xa,xb, y1 = line.split(",")  # quebra nas virgulas e retorna 3 valores
    x1.append(float(xa))
    x2.append(float(xb))
    y.append(float(y1))
dados.close()


def cria_mat_all():
    mat = np.ones((1001,3))

    for i in range(0,1001):
        mat[i][0] = x1[i]
        mat[i][1] = x2[i]
        mat[i][2] = y[i]

    mat = np.random.permutation(mat)

    return mat

def cria_centroides():
    global centroides
    mat_all = cria_mat_all()
    mat_dados = np.ones((1001,2))
    for i in range(0,1001):
        for j in range(0,2):
            mat_dados[i][j] = mat_all[i][j]

    kmeans = KMeans(n_clusters=num_neuronio_oculto, random_state=0).fit(mat_dados)

    centroides = kmeans.cluster_centers_    


def cria_mat_att_e_resp_treino_e_teste():
    global mat_att_treino, mat_resp_treino,mat_att_teste,mat_resp_teste
    mat_all = cria_mat_all()
    #TREINO
    for i in range(0,800):
        mat_att_treino[i][0] = mat_all[i][0]
    for i in range(0,800):
        mat_att_treino[i][1] = mat_all[i][1]
    
    for i in range(0,800):
        mat_resp_treino[i][0] = mat_all[i][2]

    #TESTE
    for i in range(800,1001):
        mat_att_teste[i-800][0] = mat_all[i][0]
    for i in range(800,1001):
        mat_att_teste[i-800][1] = mat_all[i][1]
    
    for i in range(800,1001):
        mat_resp_teste[i-800][0] = mat_all[i][2]    

def neuronios_ocultos():
    global centroides

    G = np.ones((800,num_neuronio_oculto+1))

    for j in range(0,800):
        for k in range(1,num_neuronio_oculto+1):
            G[j][k] = math.exp((-1)*(((mat_att_treino[j][0]- centroides[k-1][0])**2 
                                + (mat_att_treino[j][1]- centroides[k-1][1])**2)))
    return G

def neuronio_saida_W():
    global mat_resp_treino
    G = neuronios_ocultos()
    d = mat_resp_treino

    W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(G),G)),np.transpose(G)),d)
    
    return W

def testa():
    global cont, num_neuronio_oculto
    G = np.ones((201,num_neuronio_oculto+1))
   
    for j in range(0,201):
        for k in range(1,num_neuronio_oculto+1):
            G[j][k] = math.exp((-1)*(((mat_att_teste[j][0]- centroides[k-1][0])**2 
                                + (mat_att_teste[j][1]- centroides[k-1][1])**2)))


    W = neuronio_saida_W()

    resp_rede = np.dot(G,W)

    for i in range(0,201):
        if resp_rede[i][0] > 0:
            resp_rede[i][0] = 1
        else:
            resp_rede[i][0] = -1
    #print(resp_rede)
    #print((resp))
    for i in range(0,201):
        if resp_rede[i][0] == mat_resp_teste[i][0]:
            cont +=1
    
    print("Acuracia " + str((cont/201)*100) + "% " +"Quant. de amostras acertadas: " +str(cont)) 


cria_centroides()

cria_mat_att_e_resp_treino_e_teste()

testa()

