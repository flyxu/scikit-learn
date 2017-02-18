# encoding: utf-8
from numpy import *
from numpy import math

m=2
cluster=2
w_low=0.6
w_up=0.4
T=0.0005
iter=10

# 读取数据集合
def read_data():
    dataSet=[]
    fr=open('wdbc.data.txt')
    for line in fr.readlines():
        curLine=line.strip().split(',')
        fltline=map(float,curLine[2:])
        dataSet.append(fltline)
    return array(dataSet)

#求两个向量的距离
def disEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#构建初始聚类中心
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=minJ+rangeJ*random.rand(k,1)
    return array(centroids)

def gen_initial_centroids( data_content ):
    data_set_len = len( data_content )
    ori_ran = list( set( [int( data_set_len * w ) for w in random.rand( 10 * cluster )] ) )
    return [1.0 / 2.0 * plus_ndarray( array( [data_content[ori_ran[i]], data_content[ori_ran[( i + 1 ) * ( -1 )]]] ) ) for i in  range( cluster )]


#求隶属U矩阵
def u_matrix(centroids,dataSet):
    disarray1=[]
    for k in range(len(centroids)):
        disarray0=[]
        for i in range(len(dataSet)):
            disarray0.append(disEclud(dataSet[i],centroids[k]))
        disarray1.append(disarray0)
    disarray2=array(disarray1)
    x=len(disarray2[:,0])
    y=len(disarray2[0,:])
    u=zeros((x,y))
    for i in range(len(disarray2[:,0])):
        for j in range(len(disarray2[0,:])):
            u[i][j]=1.0/sum([((disarray2[i][j]/disarray2[k][j])**(2/(m-1))) for k in range(len(disarray2[:,0]))])
    return u
#根据隶属U矩阵计算不同类别分别对应的上下近似，将他存储为嵌套字典格式
def gen_BU(u_matrix):
    BU={}
    for i in range(len(u_matrix[:,0])):
        BU[i]={}
        BU[i]["up"]=[]
        BU[i]["low"]=[]
    for i in range(len(u_matrix[0,:])):
        sort_index=argsort(u_matrix[:,i])
        max_index=sort_index[-1]
        max_next_index=sort_index[-2]
        if u_matrix[:,i][max_index]-u_matrix[:,i][max_next_index]<T:
            BU[max_index]['up'].append(i)
            BU[max_next_index]['up'].append(i)
        else:
            BU[max_index]['low'].append(i)
    return BU
#重新生成新的聚类中心
# def gen_new_centroids(BU,dataSet,u_matrix):
#     for i in range(len(u_matrix[:,0])):
#         if len([j for j in BU[i]['up'] if j not in BU[i]['low']]):
#             if len(BU[i]['low'])!=0:
#                 centroids=array(w_low*sum([u_matrix[i][k]**m*dataSet[k] for k in BU[i]['low']],axis=0)/(sum([u_matrix[i][k] for k in BU[i]['low']])) \
#                 +w_up*sum([u_matrix[i][k]**m*dataSet[k] for k in BU[i]['up'] if k not in BU[i]['low']],axis=0)/(sum([u_matrix[i][k] for k in BU[i]['up'] if k not in BU[i]['low']])))
#             else:
#                 centroids=array(sum([u_matrix[i][k]**m*dataSet[k] for k in BU[i]['up'] if k not in BU[i]['low']],axis=0)/(sum([u_matrix[i][k] for k in BU[i]['up'] if k not in BU[i]['low']])))
#         else:
#             centroids=array(sum([u_matrix[i][k]**m*dataSet[k] for k in BU[i]['low']],axis=0)/(sum([u_matrix[i][k] for k in BU[i]['low']])))
#     print centroids
#     return centroids


# ndarray的相加
def plus_ndarray( ndarray_list ):
    sum_t = ndarray_list[0]
    for i in range( 1, len( ndarray_list ) ):
        sum_t += ndarray_list[i]
    return sum_t

def gen_new_centroids( BU_map, data_content, u_matrix ):
    centroids_list = []
    global w_low, w_up
    for hang in range( len( u_matrix[:, 0] ) ):
        if  len( [w for w in BU_map[hang]['up']  if w not in BU_map[hang]['low']] ) != 0:
            if len( BU_map[hang]['low'] ) != 0 :
                w_low /= sum( [u_matrix[hang][w] ** m for w in BU_map[hang]['low']] )
                w_up /= sum( [ u_matrix[hang][w] ** m for w in BU_map[hang]['up']  if w not in BU_map[hang]['low']] )
                centroids_list.append( w_low * plus_ndarray( [ u_matrix[hang][w] ** m * data_content[w] for w in BU_map[hang]['low'] ] ) \
                 + w_up * plus_ndarray( [u_matrix[hang][w] ** m * data_content[w]  for w in BU_map[hang]['up']  if w not in BU_map[hang]['low']] ) )
            else:
                centroids_list.append( plus_ndarray( [u_matrix[hang][w] ** m * data_content[w]  for w in BU_map[hang]['up']  \
                                                    if w not in BU_map[hang]['low']] ) / sum( [ u_matrix[hang][w] ** m for w in BU_map[hang]['up']  if w not in BU_map[hang]['low']] ) )
        else:
            centroids_list.append( plus_ndarray( [ u_matrix[hang][w] ** m * data_content[w] for w in BU_map[hang]['low'] ] ) / sum( [u_matrix[hang][w] ** m for w in BU_map[hang]['low']] ) )
    return centroids_list
#实现RFCM算法的主体
def rfcm(dataSet):
    centroids=gen_initial_centroids(dataSet)
    i=0
    result=[]
    while i<iter:
        u=u_matrix(centroids,dataSet)
        BU=gen_BU(u)
        centroids=array(gen_new_centroids(BU,dataSet,u))
        print type(centroids)
        i=i+1
    for k in range(len(u[0,:])):
        result.append(argsort(u[:,k])[-1])
    return result,u,centroids

#Xie-Beni指标
def xiebeni(dataSet):
    result,u,centroids=rfcm(dataSet)
    print centroids
    in_dist=0
    temp=[]
    for i in range(len(centroids)):
        for j in range(len(dataSet)):
            in_dist+=(disEclud(dataSet[j],centroids[i])**2)*(u[i][j]**m)
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if(i==j):
                continue
            else:
                temp.append(disEclud(centroids[i],centroids[j])**2)
    out_dist=len(dataSet)*min(temp)
    v=in_dist/out_dist
    return v

#  Xie-Beni index
def calc_xie_beni_index( centroids_list, u_matrix, data_content ):
    fen_zhi = sum( [sum( [ u_matrix[w][t] ** m * sum( [x * x for x in ( data_content[t] - centroids_list[w] )] )   for t in range( len( data_content ) )] )   for w in range( len( centroids_list ) )] )
    inside_centroids_list = array( [ sum( [x * x for x in ( centroids_list[i] - centroids_list[j] )] )  for i in range( len( centroids_list ) - 1 ) for j in range( i + 1, len( centroids_list ) )] )
    min_val = inside_centroids_list[argsort( inside_centroids_list )[0]]
#     print min_val,fen_zhi,inside_centroids_list
    return fen_zhi / ( len( data_content ) * min_val )




if __name__ == '__main__':
    dataSet = read_data()
    result,u,centroids=rfcm(dataSet)
    print xiebeni(dataSet)
    print calc_xie_beni_index(centroids,u,dataSet)

   #  print len(dataSet[0:])
   # # print dataSet
   #  centroids=randCent(dataSet,2)
   # # print len(centroids)
   #  ua= u_matrix(centroids,dataSet)
   #  print gen_BU(ua)
   #  print len(gen_new_centroids(gen_BU(ua),dataSet,ua))
