import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# mr=pd.read_csv('data\\mushroom.csv',header=None)
# # print(mr)
# data=[]
# label=[]
# for k,v in mr.iterrows():
#     # print('k=',k) #행인덱스
#     # print('v=\n',v)
#     label.append(v[0])
#     temp=[]
#     for i in v.iloc[1:]:
#         temp.append(ord(i))
#     data.append(temp)
# # print(label)  #정답
# # print(data)   #데이터
# #75%학습,25%검증
# trainx,testx,trainy,testy=train_test_split(data,label,train_size=0.75)
# model=RandomForestClassifier()
# model.fit(trainx,trainy)
# pred=model.predict(testx)
# print('정확도=',accuracy_score(pred,testy))
# ------------------------
# colors=[['red','green','blue','orange'],
#         ['pink','ivory','purple','tomato']]
# data=pd.DataFrame(colors,columns=['a','b','c','d'])
# print(data)
# print('-'*30)
# for d in data:
#     print(d)    #열이름
# print('-'*30)
# for k,v in data.iteritems():
#     print('k=',k)   #열이름
#     print('v=\n',v)  #값
# print('='*30)
# for k,v in data.iterrows():
#     print('k=',k)   # 행인덱스
#     print('v=\n',v)  # 값
# print('!'*30)
# for t in data.itertuples():
#     print(t)   #행인덱스와 값이 튜플로
#     print(t[0],'***')  #행인덱스
# ---------교차검증
from sklearn.svm import SVC
def split_x_y(data):
    x=[]
    y=[]
    for d in data:
        x.append(d[:4])
        y.append(d[4])
    return x,y
data=open('data\\iris.csv').read().split('\n')
# print(data)
del data[0]
# print(data)
csv=[]
for line in data:
    temp=[]
    line=line.split(',')
    print(line)  #['5.1', '3.5', '1.4', '0.2', 'setosa']
    temp.append(float(line[0]))
    temp.append(float(line[1]))
    temp.append(float(line[2]))
    temp.append(float(line[3]))
    temp.append(line[4])
    # print(temp)
    csv.append(temp)
# print(csv)
# 데이터분할
k=5
#[[],[],[],[],[]]
csvk=[[] for i in range(k)]   #[[], [], [], [], []]
# print(csvk)
for i in range(len(csv)): #i=0,1,...149
    csvk[i%k].append(csv[i])
# print(csvk)
# print(len(csvk))   #5
# print(len(csvk[0]))   #30
scores=[]
for testdata in csvk:
    print('검증용data=',testdata)   #2차원
    traindata=[]
    for i in csvk:
        if i!=testdata:
            # traindata.append(i)
            traindata=traindata+i
    # print('훈련용data=',traindata)   #2차원
    testx,testy=split_x_y(testdata)
    trainx,trainy=split_x_y(traindata)
    # print('testx=',testx)
    # print('testy=',testy)
    # 모델생성
    model=SVC()
    model.fit(trainx,trainy)
    pred=model.predict(testx)
    # print('정확도=',accuracy_score(pred,testy))
    scores.append(accuracy_score(pred,testy))
print('정확도=',scores)
print('전체정확도=',sum(scores)/len(scores))


# a=[1,2,3]
# b=['one','two','three']
# a.append(b)
# print('1번',a)
# print('2번',a+b)
print('\n\n\n\n\n\n\n\n\n')






