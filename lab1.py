
from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')

X, Y = mnist['data']*1./255, mnist['target']
num_cls = len(np.unique(Y))
hyperparams = {'hidden_neurons':200, 'learning_rate':0.01, \
               'EPOCHS':20, 'batchsize':180}
""" Initialize parameters """
parameters = {}
parameters['W1'] = np.random.normal(loc=0.,scale=np.sqrt(\
                      hyperparams['hidden_neurons']*1./len(X[0,:])),\
                      size=(hyperparams['hidden_neurons'],len(X[0,:])))
#parameters['b1'] = np.zeros(shape=(hyperparams['hidden_neurons'],))

parameters['W2'] = np.random.normal(loc=0.,scale=np.sqrt(\
                      len(np.unique(Y))*1./hyperparams['hidden_neurons']),\
                      size=(len(np.unique(Y)),hyperparams['hidden_neurons']))
#parameters['b2'] = np.zeros(shape=(len(np.unique(Y),)))

forw_prop = {}
def forward_prop(X0):
    
    forw_prop['X'] = X0
    X0 = np.dot(parameters['W1'], X0)# + parameters['b1']
    forw_prop['Z1'] = X0
    
    X0 = 1./(1. + np.exp(-X0))
    forw_prop['A1'] = X0
    
    X0 = np.dot(parameters['W2'], X0)# + parameters['b2']
    forw_prop['Z2'] = X0   
    X0 = np.exp(X0)
    X0 = X0 / np.sum(X0)
    forw_prop['A2'] = X0
    
    return X0

def gradients_mod(Y, Yhat, m):
    
    grads = {}    
    onehot = np.zeros(shape=Yhat.shape)
    onehot[Y] = 1.
    dX = Yhat - onehot
    grads['W2'] = (1./m)*np.dot(np.expand_dims(dX,1), np.expand_dims(forw_prop['A1'],0)) 
    #grads['b2'] = (1./m)*dX
    
    dA1 = np.multiply(forw_prop['A1'], (1. - forw_prop['A1']))
    dZ1 = np.multiply(np.dot(parameters['W2'].T, np.expand_dims(dX,1)), \
                      np.expand_dims(dA1, 1))
    grads['W1'] = (1./m)*np.dot(dZ1,np.expand_dims(forw_prop['X'],0))
    #print(dZ1.shape)
    #grads['b1'] = (1./m)*np.sum(dZ1,axis=1,keepdims=False)
    return grads

def loss(Yhats, m):
    return -np.sum(np.log(Yhats)) / m

losses, accuracy = [], []
batch_size, EPOCHS = hyperparams['batchsize'], hyperparams['EPOCHS'] #m
alpha = hyperparams['learning_rate'] # gradient step
numbers = np.arange(len(Y))
np.random.shuffle(numbers)
batches = [numbers[k*batch_size:k*batch_size+batch_size] for k in \
           range(int(len(numbers)/batch_size))]


for i in range(EPOCHS):
    for j0,batch in enumerate(numbers[:int(len(numbers)*0.85)]):
        cur_grad = {'W1':0,'W2':0}
        #for j in batch[:-int(len(batch)*0.15)]:
        Yhat = forward_prop(X[batch,:])
        aux_grad = gradients_mod(int(Y[batch]), Yhat, 1)#batch_size*0.85)
        cur_grad['W1'] += aux_grad['W1']
        cur_grad['W2'] += aux_grad['W2']
        parameters['W2'] -= alpha * cur_grad['W2']
        parameters['W1'] -= alpha * cur_grad['W1']
        #print(j0)
        #parameters['b2'] -= alpha * grads['b2']
        #parameters['b1'] -= alpha * grads['b1']
                        

    eval_data = [forward_prop(X[k,:])[int(Y[numbers[-int(len(numbers)*0.15)+k0]])]\
                    for k0,k in enumerate(numbers[-int(len(numbers)*0.15):])]
    losses.append(loss(eval_data,int(len(numbers)*0.15)))
    eval_data = [Y[k] for k in numbers[-int(len(numbers)*0.15):]], \
                [np.argmax(forward_prop(X[k,:]))\
                 for k0,k in enumerate(numbers[-int(len(numbers)*0.15):])]
    accuracy.append(np.sum(np.array(eval_data[0])-np.array(eval_data[1]) == 0)/int(len(numbers)*0.15))
    #if j0 % 10000 ==0:
    print(i,'epoch, loss',losses[-1],'acc',accuracy[-1])
   

plt.plot(accuracy)
plt.title('Accuracy score')
plt.grid()
plt.show()

plt.plot(losses)
plt.title('Loss function values')
plt.grid()
plt.show()
print('Final accuracy', accuracy[-1])
"""    
pred_data = np.arange(int(0.85*len(Y)), len(Y))
#np.random.randint(low=int(0.8*len(Y)),high=len(Y),size=10)
eval_data = [Y[k] for k in pred_data], [np.argmax(forward_prop(X[k,:]))\
                    for k0,k in enumerate(pred_data)]
accuracy1 = (np.sum(np.array(eval_data[0])-np.array(eval_data[1]) == 0)/(0.15*len(Y)))
print('Test accuracy', accuracy1)
#for i in pred_data:
#    Yhat = np.argmax(forward_prop(X[i,:]))
#    print('Predicted ', Yhat, ' True ', int(Y[i]))
"""
"""
plt.plot(accuracy)
plt.title('Accuracy score')
plt.grid()
plt.show()

plt.plot(losses)
plt.title('Loss function values')
plt.grid()
plt.show()
"""