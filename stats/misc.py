def get_percentile_rank(scores, your_score):
    """
    From book think_stats
    :param scores:
    :param your_score:
    :return:
    """
    count = 0
    for score in scores:
        if score <= your_score:
            count += 1

    percentile_rank = 100.0 * count / len(scores)
    return percentile_rank