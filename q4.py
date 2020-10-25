import pandas as pan
import numpy as num
from scipy.spatial import distance 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class Weather:
    import pandas as pan
    import numpy as num
    from scipy.spatial import distance 
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    def init(self):
        theta=[]
    def train(self,path):
        training=pan.read_csv(path,header=0) 
#        training=pan.read_csv('/home/srg/Desktop/2sem/SMAI/ass2/Assignment-2_Dataset/Datasets/Question-4/weather.csv',header=0)
        ansg=training.iloc[0:,4]
        training=training.drop(training.columns[[10]], axis=1)
        training=training.drop(training.columns[[4]], axis=1)
        training=training.drop(training.columns[[2]], axis=1)
        training=training.drop(training.columns[[1]], axis=1)
        training=training.drop(training.columns[[0]], axis=1)
        train=pan.DataFrame(training).to_numpy()
        scaler=MinMaxScaler()
        train=scaler.fit_transform(train)
        #training['5']=1
        
        
        #training['5']=1
#        train=pan.DataFrame(training).to_numpy()
        train=num.append(num.ones((len(train),1)),train,axis=1)
        ##print(train)
        ans=pan.DataFrame(ansg).to_numpy()
                #
        theta=num.array([[1,1,1,1,1,1,1]])
        theta=theta.T
        m=len(ans)
        predictions=train.dot(theta)
        #cost=(1/2*m)*num.sum(num.square(predictions-ans))
        
        #c1=num.zeros(100)
        #t1=num.zeros((100,2))
        
        for it in range(0,100000):
            predictions=train.dot(theta)
            w=(train.T.dot(predictions-ans))
            w=0.2*w
            w=(1/m)*w
            theta=theta-w
            # theta=theta-(1/m)*0.2*(train.T.dot(predictions-ans))
            w1=num.sum((predictions-ans)**2)
            w1=(1/2*m)*w1
            c=w1
            # (1/2*m)*num.sum(num.square(predictions-ans))
        #    print(c)
        
        print(r2_score(ans,predictions))
        self.theta=theta
    def predict(self,path):
        testing=pan.read_csv(path,header=0)
#        testing=pan.read_csv('/home/srg/Desktop/2sem/SMAI/ass2/Assignment-2_Dataset/Datasets/Question-4/weather.csv',header=0)
        testing=testing.drop(testing.columns[[9]], axis=1)
        
#        testing=testing.drop(testing.columns[[4]], axis=1)
        testing=testing.drop(testing.columns[[2]], axis=1)
        testing=testing.drop(testing.columns[[1]], axis=1)
        testing=testing.drop(testing.columns[[0]], axis=1)
        testing=pan.DataFrame(testing).to_numpy()
        scaler =  MinMaxScaler()
        testing=scaler.fit_transform(testing)
        testing=num.append(num.ones((len(testing),1)),testing,axis=1)
#        test=pan.DataFrame(testing).to_numpy()
        theta=self.theta
        predictions=testing.dot(theta)
        return predictions
