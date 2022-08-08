import argparse
import yaml
import Parser as par
import Train as tr
import Test as test


parser = par.get_parser()
p = parser.parse_args()
if p.config is not None:
    with open(p.config, 'r') as f:
        default_arg = yaml.safe_load(f)
        print(default_arg)
    key = vars(p).keys()
    for k in default_arg.keys():
        if k not in key:
            print('WRONG ARG: {}'.format(k))
            assert (k in key)
    parser.set_defaults(**default_arg)

opt = parser.parse_args()


if opt.batch_size == 1:

    print('Training stochastic ...')
    tr.train(opt)
    if opt.direct_test:
        print('Testing best model loss ...')
        test.test(opt)
        print('Testing best model ratio ...')
        opt.best_model = 'ratio'
        test.test(opt)

elif opt.batch_size > 1:
    
    print('Training in batches ...')
    tr.train(opt)
    if opt.direct_test:
        print('Testing best model loss ...')
        test.test(opt)
        print('Testing best model ratio ...')
        opt.best_model = 'ratio'
        test.test(opt)


elif  opt.batch_size < 0:
            sys.exit("Batch size is not valid.Must be > 0")
        
