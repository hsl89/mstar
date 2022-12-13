import math


def get_slopes_power_of_2(n):
    start = (2 ** (-2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * ratio ** i for i in range(n)]


# function to compute slopes for alibi bias
def get_slopes(n):
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    # In the paper, we only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    else:
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
