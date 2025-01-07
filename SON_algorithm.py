from pyspark import SparkContext
import sys, itertools, collections, operator, time, os
from datetime import datetime

# phase 1
def SON_1(iterator):
    # split baskets and define local threshhold for apriori
    all_baskets = list(iterator)
    local_support = (len(all_baskets) * min_support + num_baskets - 1) // num_baskets

    # apriori
    result = find_frequent_itemsets(all_baskets, local_support)
    return result

# phase 2
def SON_2(iterator, candidate_itemsets):
    occurrence_counts = collections.defaultdict(int)

    # create a list of baskets
    all_baskets = list(iterator)

    # get itemsets
    candidate_sets = [candidate[1] for candidate in candidate_itemsets]

    # for each basket
    for current_basket in all_baskets:
        current_basket_set = set(current_basket)
        for candidate in candidate_sets:
            # if candidate is a subset of the current basket increase count
            if candidate.issubset(current_basket_set):
                occurrence_counts[candidate] += 1

    return [(itemset, count) for itemset, count in occurrence_counts.items()]


# apriori implementation
def find_frequent_itemsets(baskets, support):
    # generate all single items
    single_itemsets = {frozenset([item]) for basket in baskets for item in basket}
    final_freq_itemsets = []

    # filter single itemsets
    curr_freq = filter_itemsets(baskets, single_itemsets, support)
    k = 1

    # loop until there are no more frequent itemsets
    while curr_freq:
        # Append the current frequent itemsets as tuples
        final_freq_itemsets.extend((k, itemset) for itemset in curr_freq)

        # generate candidates for the next size
        k += 1
        unfiltered_curr = create_candidates(curr_freq, k)

        # filter candidates based on support
        curr_freq = filter_itemsets(baskets, unfiltered_curr, support)

    return final_freq_itemsets


def create_candidates(prev_freq_itemsets, k):
    # make all possible candidates of size k
    candidates = set()
    for i in prev_freq_itemsets:
        for j in prev_freq_itemsets:
            candidate = i.union(j)
            if len(candidate) == k:
                candidates.add(candidate)

    # filter candidates by checking if all subsets of size k-1 are present in the previous frequent itemsets
    # if candidate is frequent, all subsets are frequent
    filtered_candidates = set()
    for itemset in candidates:
        all_subsets_present = True
        for subset in itertools.combinations(itemset, k - 1):
            if frozenset(subset) not in prev_freq_itemsets:
                all_subsets_present = False
                break
        if all_subsets_present:
            filtered_candidates.add(itemset)

    return filtered_candidates


def filter_itemsets(baskets, candidates, support):
    itemset_counts = {candidate: 0 for candidate in candidates}

    # count occurrences of each candidate in baskets
    for basket in baskets:
        basket_set = set(basket)
        for candidate in candidates:

            # if candidate is in basket
            if candidate.issubset(basket_set):
                itemset_counts[candidate] += 1

    # filter candidates based on support
    return {itemset for itemset, count in itemset_counts.items() if count >= support}


# writing output
def write_candidates(file, itemsets):
    # sort by size
    sorted_candidates = sorted(itemsets, key=lambda x: (x[0], sorted(x[1])))

    current_size = sorted_candidates[0][0]
    grouped_items = []

    for item in sorted_candidates:
        size = item[0]
        frozenset_items = item[1]

        # if size changes write the current group to the file
        if size != current_size:
            if current_size == 1:
                # remove parenthesis inside
                file.write(
                    ', '.join('(' + ''.join(frozenset_items) + ')' for frozenset_items in grouped_items) + '\n')
            else:
                file.write(
                    ', '.join(
                        '(' + ', '.join(sorted(frozenset_items)) + ')' for frozenset_items in grouped_items) + '\n')
            current_size = size
            grouped_items = []

        grouped_items.append(frozenset_items)

    # write remaining
    if grouped_items:
        file.write(
            ', '.join('(' + ', '.join(sorted(frozenset_items)) + ')' for frozenset_items in grouped_items) + '\n')


# reformatting for write_output
def get_sizes(itemsets):
    # include size
    return [(len(fs), fs) for fs in itemsets]


def parse(line):
    fields = line.split(',')
    transaction_date = fields[0].strip('"')
    customer_id = fields[1].strip('"')
    product_id = fields[5].strip('"')

    formatted_date = datetime.strptime(transaction_date, "%m/%d/%Y").strftime("%m/%d/%y")
    return f"{formatted_date}-{customer_id}", int(product_id)


def write_to_csv(input_file, new_input_file):
    rdd = sc.textFile(input_file)
    header = rdd.first()
    rdd = rdd.filter(lambda line: line != header)

    parsed_rdd = rdd.filter(lambda line: line != header) \
        .map(parse) \
        .filter(lambda x: x is not None)

    # write to file
    results = parsed_rdd.collect()

    with open(new_input_file, "w") as f:
        f.write("DATE_CUSTOMER_ID,PRODUCT_ID\n")  #
        for date_customer_id, product_id in results:
            f.write(f"{date_customer_id},{product_id}\n")

    return new_input_file


if __name__ == "__main__":
    start_time = time.time()

    # parse args

    filter_threshold = int(sys.argv[1])
    min_support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    # initialize sc
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    sc = SparkContext('local[*]', 'SON_Ta_Feng')
    sc.setLogLevel("WARN")

    new_input = write_to_csv(input_file, 'newinput')

    # process data
    data = sc.textFile(new_input) \
        .filter(lambda x: not x.startswith("DATE_CUSTOMER_ID")) \
        .map(lambda x: x.split(','))

    basketsRDD = data.groupByKey() \
        .mapValues(set) \
        .filter(lambda x: len(x[1]) > filter_threshold) \
        .map(lambda x: x[1]).cache()

    num_baskets = basketsRDD.count()

    # SON phase1
    candidatesRDD = basketsRDD.mapPartitions(SON_1).distinct()
    candidates = candidatesRDD.collect()

    # SON phase 2
    resultsRDD = basketsRDD.mapPartitions(lambda it: SON_2(it, candidates)) \
        .reduceByKey(operator.add) \
        .filter(lambda x: x[1] >= min_support) \
        .map(lambda x: x[0])

    frequent_itemsets = resultsRDD.collect()
    frequent_itemsets = get_sizes(frequent_itemsets)

    # write output
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")
        write_candidates(f, candidates)

        f.write("\nFrequent Itemsets:\n")
        write_candidates(f, frequent_itemsets)

    end_time = time.time()

    # print duration
    print("Duration : " + str(end_time - start_time))