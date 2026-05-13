def fm_interaction(stacked):
    sum_square = stacked.pow(2).sum(dim=1)
    square_sum = stacked.sum(dim=1).pow(2)
    return 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
