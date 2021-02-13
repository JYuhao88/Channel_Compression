import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--square", type=int,
                    help="display a square of a given number")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("-p")
args = parser.parse_args()
# answer = args.s**2
# if args.verbose:
#     print("the square of {} equals {}".format(args.square, answer))
# else:
#     print(answer)
print(args.p)