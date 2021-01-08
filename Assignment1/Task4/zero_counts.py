import itertools

def gen_sentences(vocabulary, M):
    return itertools.product(*[vocabulary]*M)

def gen_bigrams(sentence):
    sentence = ["BOS"] + [*sentence]
    for i in range(0, len(sentence)-1):
        yield sentence[i] + " " + sentence[i+1]

def gen_possible_bigrams(vocabulary):
    for w1 in ["BOS"] + vocabulary:
        for w2 in vocabulary:
            yield w1 + " " + w2

vocabulary = ["a", "b", "c", "d"]
for M in range(20):

    #print()
    #print(f"Vocabulary: {len(vocabulary)+1}")
    #print("  BOS", " ".join(vocabulary))

    possible_bigrams = list(gen_possible_bigrams(vocabulary))
    #print()
    #print(f"All possible Bigrams: {len(possible_bigrams)}")
    #print(" ", ", ".join(possible_bigrams))

    counts = {}
    sentences = list(gen_sentences(vocabulary, M))
    #print()
    #print(f"Sentences: {len(sentences)}")
    for sentence in gen_sentences(vocabulary, M):
        #print(" ".join(sentence))
        bigrams = list(gen_bigrams(sentence))
        unique = len(set(bigrams))
        #print(" ", f"Bigrams: {len(bigrams)}, Unique: {unique}")
        #print("  ", ", ".join(bigrams))
        if unique not in counts:
            counts[unique] = 1
        else:
            counts[unique] += 1


    total = sum(counts.values())
    expected = sum(map(lambda x:x[0]*x[1], counts.items()))/total
    v = len(vocabulary)
    a = v**2
    print()
    print("M =",M)
    print("len(vocabulary) =",len(vocabulary))
    print(f"Expected unique: {expected}")
    print(f"Calculated unique: {a-a*((a-1)/a)**(M-1)+1}")
    
    for count, occurences in counts.items():
        print(" ", f"{occurences} sentences had {count} unique bigrams.")

