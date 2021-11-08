import argparse

args = None
test_args = 10

def init_args():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data_div', type=int, default=5, metavar='S',
                        help='how many dataset for divide (default: 5)')
    parser.add_argument('--val_data_num', type=int, default=0, metavar='S',
                        help='which dataset will be choose as Validation data (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--gpunum', type=int, default=0,
                        help='number of gpu use')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--log', type=str, default=None,
                        help='If log file name given, we write Logs')
    parser.add_argument('--W', type=int, default=6,
                        help='Row Size')
    parser.add_argument('--model', type=int, default=1,
                        help='Select Model number')
    parser.add_argument('--test', type=str, default=None,
                        help='Do the Test')


    global args
    args = parser.parse_args()

    import sys

    if args.val_data_num >= args.data_div:
        print("Bigger val_data_num than data_div", file=sys.stderr)
        sys.exit()

    print(args)
