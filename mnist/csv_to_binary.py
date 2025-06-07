import pandas as pd


def csv_to_binary(input_csv, output_file):
    df = pd.read_csv(input_csv)
    # limit to 1000 rows for testing
    df = df.head(1000)
    size = df.shape[0]
    with open(output_file, 'wb') as f:
        f.write(size.to_bytes(4, byteorder='big'))
        for index, row in df.iterrows():
            # Write first byte as the label in big endian format
            f.write(row[0:1].to_numpy().tobytes())
            f.write(row[1:].to_numpy().tobytes())


if __name__ == "__main__":
    csv_to_binary("train.csv", "train.bin")
    csv_to_binary("test.csv", "test.bin")
