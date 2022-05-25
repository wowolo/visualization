def report_hyperparam(key, **kwargs):
    print('The following hyperparameters have been used for the construkction of the {} \
        architecture:'.format(key))
    for key in kwargs.keys():
        print('{}: {}'.format(key, kwargs[key]))
    