import argparse


class StoreDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):

        kv = {}
        if not isinstance(values, (list,)):
            values = (values,)
        for value in values:
            n, v = value.split(':')

            # allow for int/float conversion
            converted = False
            if not converted:
                try:
                    v = int(v)
                    converted = True
                except ValueError:
                    pass
            if not converted:
                try:
                    v = float(v)
                    converted = True
                except ValueError:
                    pass

            kv[n] = v
        
        if len(kv) == 0:
            kv = None

        setattr(namespace, self.dest, kv)


parser = argparse.ArgumentParser()
parser.add_argument('--experimentbatch_name', type=str)
parser.add_argument('--config_trainer', action=StoreDict, nargs='*')

args = parser.parse_args()

print(args)
print(args.config_trainer)