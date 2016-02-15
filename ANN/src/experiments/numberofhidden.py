'''
Created on Feb 14, 2016

@author: Wuga
'''
import network
import data
import visualization
import matplotlib.pyplot as plt

initial_setting = {
    "inputs_N"          : 2,       
    "layers"            : [ (100, network.ReLU),(1, network.Sigmoid)],
    "weights_L"         : -0.1,     
    "weights_H"         : 0.1,      
    "save"              : False, #haven't implement this one yet  
    "loss"              : network.CrossEntropyLoss, 
    "learning_C"        : [],
    "testing_C"         : []
                   }

def replace_value_with_definition(key_to_find, definition):
    for key in initial_setting.keys():
        if key == key_to_find:
            initial_setting[key] = definition           

dataset = data.load()
features = dataset["train_data"]
targets = dataset["train_labels"]
test_features = dataset["test_data"]
test_targets = dataset["test_labels"]
m,n = features.shape
replace_value_with_definition("inputs_N",n)
replace_value_with_definition("weights_L",-1.0/n)
replace_value_with_definition("weights_H",1.0/n)
print initial_setting
hiddensettings = [(10,'g'), (50,'b'),(100,'r')]

results = []
for hidden in hiddensettings:
    initial_setting["layers"][0]=(hidden[0],network.ReLU)
    NN = network.NetworkFrame(initial_setting)
    features_normalized,mean,std = data.normalize(features)
    test_normalized,_,_ = data.normalize(test_features,mean,std)
    NN.Train(features_normalized, targets, test_normalized, test_targets, 10e-5, 100, 200, 0.001)
    testing_record = NN.GetTestingRecord()
    testing_indecs = [x[0] for x in testing_record]
    testing_errors = [x[1] for x in testing_record]
    results.append([testing_indecs,testing_errors,hidden[1]])

plot_config = []
legends= []
for i in range(len(results)):
    x,=plt.plot(results[i][0],results[i][1])
    legends.append(x)
plt.legend(legends,['10','50','100'])
plt.title('Test error curve of different number of hidden node')
plt.show()
    