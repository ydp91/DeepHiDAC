import argparse
import datetime
import dateutil
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSFM")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--out_dir", dest="out_dir", type=str, default="./out/")
    parser.add_argument("--model_dir", dest="model_dir", type=str, default="./model/")
    parser.add_argument("--dataset", dest="dataset", type=str, default="raw")
    parser.add_argument("--D_dict", dest="D_dict", type=str, default="./model/D_raw_100.pth")
    parser.add_argument("--C_dict", dest="C_dict", type=str, default="./model/C_raw_100.pth")
    parser.add_argument("--O_dict", dest="O_dict", type=str, default="./model/O_raw_100.pth")
    parser.add_argument("--V_dict", dest="V_dict", type=str, default="./model/VAE_raw_50.pth")
    parser.add_argument("--traj_len", dest="traj_len", type=int, default=9)
    parser.add_argument("--delim", dest="delim", default=",")
    parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate") #cvae 1e-4
    parser.add_argument("--lr_step", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.8)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if "timestamp" not in args:
        args.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )
    return args


if __name__ == "__main__":
    print(parse_args())
