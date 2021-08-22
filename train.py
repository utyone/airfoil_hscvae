import argparse
import os
import sys
import json
from util import train


def get_parameter(path, latent_dim):
    with open(path) as f:
        p = json.load(f)
    if latent_dim:
        p["n_z"] = latent_dim
    return p


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Parser
    parser = argparse.ArgumentParser(description='This script is ...', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('model', action='store', nargs=None, const=None, default=None, type=str, choices=None,
                        metavar=None, help="""Name of model to use. 
- ncvae: normal conditional variational autoencoder\n- hscvae: hyperspherical conditional vae """)
    parser.add_argument('-n', '--latent_dim', action='store', nargs='?', const=None, default=4, type=int,
                        choices=None, help='Dimension of latent vector. [default: 20]', metavar=None)
    parser.add_argument('-b', '--batch_size', action='store', nargs='?', const=None, default=100, type=int,
                        choices=None, help='Batch size. [default: 100]', metavar=None)
    parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                        choices=None, help='Epoch number. [default: 150]', metavar=None)
    parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                        choices=None, help='Learning rate. [default: 0.005]', metavar=None)
    parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                        choices=None, help='Gradient clipping. [default: None]', metavar=None)
    args = parser.parse_args()
    print("\n Start train %s \n" % args.model)
    save_path = "./log/%s_%i/" % (args.model, args.latent_dim)
    param = get_parameter("./parameter/%s.json" % args.model, args.latent_dim)
    opt = dict(network_architecture=param, batch_size=args.batch_size, learning_rate=args.lr, save_path=save_path,
               max_grad_norm=args.clip, latent=args.latent_dim)

    if args.model == "ncvae":
        from model import HSCvae as Model
        _mode, _inp_img = "conditional", False
    elif args.model == "hscvae":
        from model import NCvae as Model
        _mode, _inp_img = "conditional", False
    else:
        sys.exit("unknown model !")

    if _mode == "conditional":
        opt["label_size"] = 1
    print(Model.__doc__)
    model = Model(**opt)
    lat = args.latent_dim
    train(model=model, lat=lat, epoch=args.epoch, save_path=save_path, mode=_mode, input_image=_inp_img)


