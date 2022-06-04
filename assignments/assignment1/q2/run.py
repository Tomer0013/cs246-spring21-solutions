import itertools

if __name__ == '__main__':

    #### d ####

    # A-priori pass 1
    product_count = {}
    for row in open('data/browsing.txt', encoding='utf8'):
        for product in row.split():
            try:
                product_count[product] += 1
            except KeyError:
                product_count[product] = 1

    relevant_products_dict = {k:v for k,v in product_count.items() if v >= 100}

    # A-priori pass 2
    product_pairs_count = {}
    for row in open('data/browsing.txt', encoding='utf8'):
        relevant_products = sorted(list(set([product for product in row.split() if product in relevant_products_dict])))
        for i, prod_1 in enumerate(relevant_products):
            for prod_2 in relevant_products[i+1:]:
                try:
                    product_pairs_count[(prod_1, prod_2)] += 1
                except KeyError:
                    product_pairs_count[(prod_1, prod_2)] = 1

    freq_pairs = {k:v for k,v in product_pairs_count.items() if v >= 100}

    # Create rules
    assoc_rules_dict = {}
    for pair in freq_pairs.keys():
        a, b = pair
        support_a = relevant_products_dict[a]
        support_b = relevant_products_dict[b]
        assoc_rules_dict[(a, b)] = freq_pairs[pair] / support_a
        assoc_rules_dict[(b, a)] = freq_pairs[pair] / support_b

    # Print top 5 rules
    print('Top 5 pairs rules:')
    rules_list = sorted(list(assoc_rules_dict.items()), key=lambda x: (-x[1], x[0][0]))
    for rule in rules_list[:5]:
        print(f'{rule[0][0]} ---> {rule[0][1]} [Conf={rule[1]:.3f}]')

    ### e ###

    # A-priori pass 3
    product_triples_count = {}
    for row in open('data/browsing.txt', encoding='utf8'):
        relevant_pairs = [subset for subset in itertools.combinations(row.split(), 2) if subset in freq_pairs]
        relevant_products = list(set([product for pair in relevant_pairs for product in pair]))
        for pair in relevant_pairs:
            for product in relevant_products:
                if product not in pair:
                    try:
                        product_triples_count[(*pair, product)] += 1
                    except KeyError:
                        product_triples_count[(*pair, product)] = 1

    freq_triples = {k:v for k,v in product_triples_count.items() if v >= 100}

    # Create rules
    assoc_rules_triples_dict = {}
    for triple in freq_triples.keys():
        x, y, z = triple
        assoc_rules_triples_dict[((x, y), z)] = freq_triples[triple] / freq_pairs[(x, y)]
        assoc_rules_triples_dict[((x, z) if x < z else (z, x), y)] = freq_triples[triple] / freq_pairs[(x, z) if x < z else (z, x)]
        assoc_rules_triples_dict[((y, z) if y < z else (z, y), x)] = freq_triples[triple] / freq_pairs[(y, z) if y < z else (z, y)]

    # Print top 5 rules
    print('\nTop 5 triples rules:')
    rules_list = sorted(list(assoc_rules_triples_dict.items()), key=lambda x: (-x[1], x[0][0][0], x[0][0][1]))
    for rule in rules_list[:5]:
        print(f'{rule[0][0]} ---> {rule[0][1]} [Conf={rule[1]:.3f}]')
