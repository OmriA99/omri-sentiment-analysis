import csv
import numpy as np

def show_some_stats(filepath):
    with open(filepath, mode='r') as train:
        csv_reader = csv.DictReader(train, delimiter="\t")

        avg_len = 0
        minLen = 100000
        maxLen = -1
        line_count = 0
        under_10 = 0
        b_50_100 = 0
        label_count = np.zeros([5])

        print(f'[show_stats()], fieldnames = {csv_reader.fieldnames}')
        print("==================")
        for row in csv_reader:

            l = len(row['Phrase'])
            if l < 10:
                under_10 += 1
            if l >= 30 and l <= 80:
                b_50_100 += 1

            avg_len += l
            minLen = min(minLen, l)
            maxLen = max(maxLen, l)
            label_count[int(row['Sentiment'])] += 1
            line_count += 1


        avg_len /= line_count

        print("Avarege length of a phrase is {} with max = {} and min = {}".format(avg_len, maxLen, minLen))
        print(f'There are {under_10} phrases with length under 10 and {b_50_100} between 30 and 80')
        print(f'The label count is {label_count}')
        print(f'Processed {line_count} lines.')

def delete_unwanted(minLen, maxLen, maxLabelCount):
    with open('pp_data.tsv', 'w') as out, open('data/train.tsv', 'r') as train:
        trainReader = csv.DictReader(train, delimiter="\t")
        fieldnames = trainReader.fieldnames
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")

        print(f'[delete_unwanted()], fieldnames = {writer.fieldnames}')
        writer.writeheader()

        lc = 0
        label_counter = np.zeros([5])
        for row in trainReader:
            l = len(row['Phrase'])
            label = int(row['Sentiment'])

            if l >= minLen and l <= maxLen and label_counter[label] < maxLabelCount:
                label_counter[label] += 1
                writer.writerow(row)

            lc += 1


if __name__ == "__main__":
    delete_unwanted(30, 80, 3000)
    show_some_stats('pp_data.tsv')