# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:17:47 2018

@author: benji
"""

import scipy
from PIL import Image
from scipy import ndimage
#import imageio
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.manifold
import matplotlib.pyplot as plt
import random as rd

path='C:\\Users\\benji\\Desktop\\Cours\\GMML\\'
img_path=path+'\\Images\\train\\'
X=[]
Y=[]
i=0
for dirname , dirnames , filenames in os.walk(img_path):
    print(dirname,dirnames,filenames)
    for filename in os.listdir(dirname):
        print(i)
#        if i>=50:
#            break
        try:
            temp=Image.open(os.path.join(dirname , filename))
#            if im.mode!='RGB':
#                if im.mode!='RGBA':
#                    print(i,im.mode,filename)
#                im=im.convert('RGB')
            im=temp.convert('L')
            im=im.resize((50,50))
            im=np.asarray(im)
            im=im.flatten() #RGB(0,0)RGB(0,1)...RGB(0,100)RGB(1,0)...RGB(100,100)
            X.append(im)
            Y.append(os.path.join(dirname , filename)[len(img_path):-4])
            i+=1
        except:
            pass


df=np.array(X) #one line by image, 3 column by pixels
df=pd.DataFrame(df)
df.to_csv(path+'images.csv',header=None,index=False)
pd.Series(Y).to_csv(path+'id images.csv',header=None,index=False)

cov=df.cov()
cov.to_csv(path+'cov.csv',header=None)
inv_cov = pd.DataFrame(np.linalg.pinv(cov.values))
inv_cov.to_csv(path+'inv_cov.csv',header=None)
sqrt_inv_cov=scipy.linalg.sqrtm(inv_cov)
pd.DataFrame(sqrt_inv_cov).to_csv(path+'sqrt_inv_cov.csv',header=None,index=False)

maha_df=df.dot(pd.DataFrame(sqrt_inv_cov))
maha_df.to_csv(path+'maha_df.csv',header=None,index=False)
#maha_dist=scipy.spatial.distance_matrix(maha_df,maha_df) #Mahalanobis distance matrix
#euc_dist=scipy.spatial.distance_matrix(df,df) #euclidean distance matrix

N=maha_df.shape[0]
samp_id=rd.sample(range(N),8000)
pd.Series(samp_id).to_csv(path+'samp ISOMAP.csv',header=None,index=False)
to_isomap=maha_df[maha_df.index.isin(samp_id)]

for var in [X,cov,inv_cov,im]: #cleaning
    del var

dim_reduced=2
reducer=sklearn.manifold.Isomap(n_components=dim_reduced,n_jobs=-1)
reduced=reducer.fit_transform(to_isomap)


temp=pd.DataFrame(reduced)
temp.to_csv('reduced by isomap.csv',header=None,index=False)
temp.index=[Y[i] for i in samp_id]
temp.reset_index(inplace=True)
info=pd.read_csv(path+'\\train_info.csv\\train_info.csv')
info['filename']=info['filename'].str.slice(stop=-4).astype(int)
temp=temp.merge(info,how='left',left_on='index',right_on='filename')
plt.scatter(temp[0],temp[1])

temp.to_csv(path+'resultats isomap.csv',index=False)
import itertools
from matplotlib import markers

m_styles = markers.MarkerStyle.markers
temp['genre']=temp['genre'].fillna('nan')
genres=temp['genre'].unique()
genres.sort()
nb_genres=len(genres)
colormap = plt.cm.Dark2.colors   # Qualitative colormap
for i,(marker,color) in zip(range(nb_genres),itertools.product(m_styles, colormap)):
    plt.scatter(temp[temp['genre']==genres[i]][0],temp[temp['genre']==genres[i]][1], color=color,marker=marker,label=genres[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,ncol=4)
plt.savefig(path+'Plot ISOMAP.png')

fig = plt.figure(1)
ax = fig.add_subplot(111)
for i,(marker,color) in zip(range(nb_genres),itertools.product(m_styles, colormap)):
    ax.scatter(temp[temp['genre']==genres[i]][0],temp[temp['genre']==genres[i]][1], color=color,marker=marker,label=genres[i])
lgd=ax.legend(bbox_to_anchor=(-0.3, 1.8), loc=2, borderaxespad=0.,ncol=4)
fig.savefig(path+'Plot ISOMAP3.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


#dim_reduced=10
##dist=euc_dist
#reducer=sklearn.manifold.Isomap(n_components=dim_reduced)
#reducer.fit(df)
#reduced=reducer.transform(df)


import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

min_maha=maha_df.min().reshape(1,-1) #for rescaling
max_maha=maha_df.max().reshape(1,-1)

    
indexes=np.array(temp['index'])
prop_train=0.8
N=reduced.shape[0]
id_train=rd.sample(set(indexes),int(N*prop_train))
df_reduced=temp[[0,1]].copy()
df_reduced.index=indexes
maha_df.index=Y
train=df_reduced[df_reduced.index.isin(id_train)].sort_index()
train_labels=maha_df[maha_df.index.isin(id_train)].sort_index()
id_test=[i for i in indexes if i not in id_train]
test=df_reduced[~df_reduced.index.isin(id_train)].sort_index()
test_labels=maha_df[maha_df.index.isin(id_test)].sort_index()
#defining how to create a neural network
class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

def make_batches(N_total, N_batch):
    start = 0
    batches = []
    while start < N_total:
        batches.append(slice(start, start + N_batch))
        start += N_batch
    return batches

def make_nn_funs(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        pred=predictions(W_vect, X)
        pred=(pred*(max_maha+min_maha)+max_maha-min_maha)/2 #rescaling after tanh
        dist = np.mean((pred-T)**2)
        return - log_prior + dist

    def frac_err(W_vect, X, T):
        pred=predictions(W_vect, X)
        pred=(pred*(max_maha+min_maha)+max_maha-min_maha)/2 #rescaling after tanh
        dist = np.mean((pred-T)**2)
        return dist

    return parser.N, predictions, loss, frac_err

class full_layer(object):
    def __init__(self, size,nonlinearity):
        self.size = size
        self.nonlinearity=nonlinearity

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_size, self.size))
        self.parser.add_weights('biases', (self.size,))
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)

# process data
reshaping = lambda x : x.reshape((x.shape[0], 1, x.shape[1]))
train, train_labels, test, test_labels = (
        np.array(train),np.array(train_labels), np.array(test), np.array(test_labels))
train = reshaping(train) 
test  = reshaping(test)  
N_data = train.shape[0]

#cross-validation
test_perf_optim=10**12
for L2_reg in [0]+[10**i for i in range(-4,1)]: #regularization L2
    for neurone_number in [3,5,10,30,50]: #number of neurones in intermediate layer
        for learning_rate in [0.05,0.001]:
            print('NEW PARAMETERS')
            print(L2_reg,neurone_number,learning_rate)
            # Make neural net functions
            # Network parameters
            input_shape = (1, dim_reduced)
            layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
                           full_layer(train_labels.shape[1],lambda x : np.tanh(x))]
            # Training parameters
            param_scale = 0.1
            momentum = 0.9
            batch_size = 256
            num_epochs = 50
            
            N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)
            loss_grad = grad(loss_fun)
            
            # Initialize weights
            rs = npr.RandomState()
            W = rs.randn(N_weights) * param_scale
            
            print("    Epoch      |    Train err  |   Test error  ")
            def print_perf(epoch, W):
                test_perf  = frac_err(W, test, test_labels)
                train_perf = frac_err(W, train, train_labels)
                print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))
            
            # Train with sgd
            batch_idxs = make_batches(N_data, batch_size)
            
            cur_dir = np.zeros(N_weights)
            
            for epoch in range(num_epochs):
                print_perf(epoch, W)
                for idxs in batch_idxs:
                    grad_W = loss_grad(W, train[idxs], train_labels[idxs])
                    cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
                    W -= learning_rate * cur_dir
            
            test_perf  = frac_err(W, test, test_labels)
            if test_perf<test_perf_optim: #updating optimal paramaters
                print('NEW OPTIMUM')
                print(L2_reg,neurone_number,learning_rate)
                test_perf_optim=test_perf
                params_optim=L2_reg,neurone_number,learning_rate
                W_optim=W
                
            
print(params_optim) #(0, 50, 0.05)
print(test_perf_optim) #48.936731155351914
#saving results
pd.DataFrame(np.array(params_optim),index=['L2_reg','neurone_number','learning_rate']).to_csv(path+'params_optim coordonnees.csv',header=False)
pd.DataFrame(W_optim,).to_csv(path+'W_optim coordonnees.csv',header=False)

L2_reg,neurone_number,learning_rate=params_optim
layer_specs = [full_layer(neurone_number,lambda x : np.tanh(x)),
               full_layer(train_labels.shape[1],lambda x : np.tanh(x))]
N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(input_shape, layer_specs, L2_reg)

# checking error
pred=pred_fun(W_optim,test)
pred=pred.dot(sqrt_inv_cov).dot(cov)
pred=pred.astype(int)
pred=pd.DataFrame(pred)
for col in pred.columns:
    pred[col]=pred[col].apply(lambda x : min(max(x,0),255))
pred=np.array(pred)

temp=df_reduced[~df_reduced.index.isin(id_train)].sort_index()
ids=temp.index
for n in range(len(id_test)):
    if n<10:
        i=ids[n]
        print(i)
        im=Image.fromarray(pred[n].reshape((50,50)),"L")
        im.save(path+"results\\nouvelle image {}.jpg".format(i))
    





