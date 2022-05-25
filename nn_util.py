def report_hyperparam(key, *args):
    print('The following hyperparameters have been used for the construkction of the {} \
        architecture:'.format(key))
    for i in len(args):
        hparam = args[i]
        print('{}: {}'.format(str(hparam), hparam))