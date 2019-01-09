import os
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance

import numpy as np
import pdb

def main(config, resume, datatype):
    if datatype == 'valid':
        data_loader = get_instance(module_data, 'valid_data_loader', config)
        USE_LABEL   = True
        # # setup data_loader instances
        # data_loader = getattr(module_data, config['data_loader']['type'])(
        #     config['data_loader']['args']['data_dir'],
        #     batch_size=512,
        #     shuffle=False,
        #     validation_split=0.0,
        #     training=False,
        #     num_workers=2
        # )
    elif datatype == 'test':
        config['test_data_loader']['num_workers']   = 4
        data_loader = get_instance(module_data, 'test_data_loader', config)
        USE_LABEL   = False
    else:
        raise NotImplementedError

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    if USE_LABEL:
        total_loss  = 0.0
        total_metrics = torch.zeros(len(metric_fns))
    else:
        pred_y_list = []

    with torch.no_grad():
        if USE_LABEL:
            for i, (data, target) in enumerate(data_loader):
                if i % 100 == 0:
                    print("{}/{} start".format(i, len(data_loader)))
                data, target = data.to(device), target.to(device)
                output = model(data)
                #
                # save sample images, or do something with output here
                #
                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size
        else:
            for i, data in enumerate(data_loader):
                if i % 100 == 0:
                    print("{}/{} start".format(i, len(data_loader)))
                data    = data.to(device)
                output  = model(data)
                i_pred_y= torch.argmax(output,dim=1).tolist()
                pred_y_list.extend(i_pred_y)
    if USE_LABEL:
        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        print(log)
    else:
        class_num   = config['arch']['args']['num_classes']
        test_res_path   = resume + '_test_res.csv'
        test_res    = np.zeros((len(pred_y_list), class_num), dtype=np.int32)
        for i, i_pred_y in enumerate(pred_y_list):
            test_res[i, i_pred_y]   = 1
        np.savetxt(test_res_path, test_res, fmt='%d', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--datatype', dest='datatype', \
        help='datatype: valid or test', default='valid', type=str)

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device

    main(config, args.resume, args.datatype)
