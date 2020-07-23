import json
import matplotlib.pyplot as plt
import numpy as np

Data_Dir = './measurements'
Optimizers = ['Adagrad', 'Adam', 'HB', 'RMSProp', 'SGD']
Array_measurements = {}

for i in Optimizers:
    measurements = []
    for j in range(1,6):
        num = str(j + int ('0'))
        read_file = open(Data_Dir+'/'+i+'_'+num+'.json', 'r')
        string_dict= json.load(read_file)
        string_dict = string_dict.replace("\'", "\"")
        dictionary = json.loads(string_dict)
        measurements.append(dictionary)
    loss = [measurements[j]['loss'] for j in range(5)]
    min_loss = [min(idx) for idx in zip(*loss)]
    max_loss = [max(idx) for idx in zip(*loss)]
    mean_loss = np.divide([sum(idx) for idx in zip(*loss)],5)

    val_loss = [measurements[j]['val_loss'] for j in range(5)]
    min_val_loss = [min(idx) for idx in zip(*val_loss)]
    max_val_loss = [max(idx) for idx in zip(*val_loss)]
    mean_val_loss = np.divide([sum(idx) for idx in zip(*val_loss)],5)

    acc = [measurements[j]['accuracy'] for j in range(5)]
    min_acc = [min(idx) for idx in zip(*acc)]
    max_acc = [max(idx) for idx in zip(*acc)]
    mean_acc = np.divide([sum(idx) for idx in zip(*acc)],5)

    val_acc = [measurements[j]['val_accuracy'] for j in range(5)]
    min_val_acc = [min(idx) for idx in zip(*val_acc)]
    max_val_acc = [max(idx) for idx in zip(*val_acc)]
    mean_val_acc = np.divide([sum(idx) for idx in zip(*val_acc)],5)

    Array_measurements[i]={
        'loss': mean_loss,
        'val_loss': mean_val_loss,
        'acc': mean_acc,
        'val_acc': mean_val_acc,
    }



t = np.arange(0, 250, 1)
fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(t,Array_measurements['SGD']['loss'],c='g',ls='-',label='SGD')
ax.plot(t,Array_measurements['HB']['loss'],c='k',ls='-',label='SGD with momentum')
ax.plot(t,Array_measurements['Adagrad']['loss'],c='b',ls='-',label='AdaGrad',fillstyle='none')
ax.plot(t,Array_measurements['RMSProp']['loss'],c='r',ls='-',label='RMSProp')
ax.plot(t,Array_measurements['Adam']['loss'],ls='-',label='Adam',fillstyle='none')

plt.title("Training Loss")
plt.ylabel('Loss')  
plt.xlabel('Epochs')

plt.legend()
plt.show()



        