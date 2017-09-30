def argmax(fun, states):
    """
    Returns the state for which fun is maximized
    :param fun:
    :param states:
    :return:
    """
    import operator
    func_tuples = ((state, fun(state)) for state in states)
    return max(func_tuples, key=operator.itemgetter(1))[0]
