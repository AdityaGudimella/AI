"""
Implements classes that represent the action spaces (domains) and state spaces (domains)
"""

import numpy as np


class Domain:
    """
    Represents a continuous domain
    
    Just wraps around the Range class by adding the sample method to it. Any method that works with Range objects
    will work with Domain objects.
    """

    def __init__(self, domain, boundaries=None):
        """
        
        :param domain: can be a Ranger.Range object or an iterable of len = 2. 
        :param boundaries: if range is not Ranger.Range object then boundaries is an iterable of len 2, with each 
        element being one of the two strings ['open', 'closed']
        """
        import operator
        if not isinstance(domain, Range):
            assert len(domain) == 2 and len(boundaries) == 2
            boundaries = ''.join([boundaries[0].lower(), boundaries[1].lower().capitalize()])
            self.domain = operator.methodcaller(boundaries, *domain)(Range)
        else:
            self.domain = domain
        self.dtype = type(self.domain.lowerEndpoint())

    def contains(self, value):
        try:
            return self.domain.contains(value)
        except ValueError:
            return self.domain.contains(self.dtype(value))

    def __getattr__(self, item):
        # Todo: This is a temporary hack. Replace this with proxy methods instead.
        return getattr(self.domain, item)

    def sample(self, size=1):
        """
        Returns randomly selected values from the domain.
        :param size: Size of the sample. Can be a tuple, in which case an ndarray of that shape will be returned
        :return: ndarray
        """
        if isinstance(self.domain.lowerEndpoint(), int):
            low = self.domain.lowerEndpoint()
            if not self.domain.contains(self.domain.lowerEndpoint()):
                low = self.domain.lowerEndpoint() + 1
            if self.domain.contains(self.domain.upperEndpoint()):
                high = self.domain.upperEndpoint() + 1
            else:
                high = self.domain.upperEndpoint()
            return np.random.randint(low=low, high=high, size=size)
        elif isinstance(self.domain.lowerEndpoint(), float):
            return np.random.uniform(self.domain.lowerEndpoint(), self.domain.upperEndpoint(), size=size)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    domain = Domain(Range.open(1, 5))
