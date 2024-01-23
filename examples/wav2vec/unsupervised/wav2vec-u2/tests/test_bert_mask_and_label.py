

import random


def random_word(self, sentence):
    tokens = sentence.split()
    output_label = []
    unk_index =1
    mask_index = 4
    vocab=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(len(vocab))

            # 10% randomly change token to current token
            else:
                tokens[i] = unk_index

            output_label.append(unk_index))

        else:
            tokens[i] = unk_index
            output_label.append(0)

    return tokens, output_label

if __name__=="__main__":
    
