def report_hyperparam(key, hyperparam_dict):
    print('The following hyperparameters have been used for the construkction of the {} \
        architecture:'.format(key))
    for paramkey in hyperparam_dict.keys():
        print('{}: {}'.format(paramkey, hyperparam_dict[paramkey]))
    