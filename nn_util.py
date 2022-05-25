from torch.utils.data import TensorDataset, DataLoader

def report_hyperparam(key, hyperparam_dict):

    print('The following hyperparameters have been used for the construkction of the {} \
        architecture:'.format(key))
        
    for paramkey in hyperparam_dict.keys():
        print('{}: {}'.format(paramkey, hyperparam_dict[paramkey]))
    


def DataGenerator(x_train, y_train, **kwargs):

    tensor_x = torch.Tensor(x_train).double()
    tensor_y = torch.Tensor(y_train).double()

    dataset = TensorDataset(tensor_x, tensor_y)
    data_generator =  DataLoader(dataset, **kwargs)

    return data_generator