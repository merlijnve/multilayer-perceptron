import sys
from mlp.support_functions import read_cancer_dataset


def main():
    try:
        if len(sys.argv) != 3:
            print("Usage: python3 split.py <filename> <test size in percent>")
            exit(1)

        cancer_data = read_cancer_dataset(
            sys.argv[1], categorize=False)

        test_size = int(int(sys.argv[2]) / 100 * len(cancer_data))
        print("Data size %d - Test size: %d" %
              (len(cancer_data) - test_size, test_size))
        with open(sys.argv[1] + "_test", 'w') as test_data:
            for line in cancer_data[:test_size]:
                test_data.write(','.join(map(str, line)) + '\n')
            test_data.close()
        with open(sys.argv[1] + "_train", 'w') as train_data:
            for line in cancer_data[test_size:]:
                train_data.write(','.join(map(str, line)) + '\n')
            train_data.close()

    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
