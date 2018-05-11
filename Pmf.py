"""This file contains class definitions for:

Hist: represents a histogram (map from values to integer frequencies).
Pmf: represents a probability mass function (map from values to probs).
_DictWrapper: private parent class for Hist and Pmf.

* 요약 통계는 데이터를 간결하게 표현해 준다는 이점이 있지만 이것만을 의지하면 위험하다.
* 실제 데이터를 올바르게 이해하기 위해서는 데이터의 분포(distribution)을 보아야 하다.
* 일반적으로 분포란 데이터 값들이 어떤 값을 많이 갖는지를 나타낸 것이다.

* 분포를 표현하기 위해 가장 널리 쓰이는 방법은
* 각 데이터 값들의 빈도나 확률값을 표현한 그래프인 히스토그램(Histogram)
* 빈도(frequency)란 데이터셋 내에서 각 값들이 몇 개 존재하는지를 나타낸 것.
* 확률(probability)이란 각 빈도값을 표본수 n으로 나누어 표현한 값.
* 빈도값에서 확률값을 구하려면 각 값에 n을 나눠 주면 된다. -> 정규화(normalization)
* 정규화된 히스토그램을 확률질량함수(Probability Mass Function, PMF)라 한다.
* 확률질향함수란 각 값들을 확률로 변환하는 함수를 의미한다.
"""

import logging
import math
import random
from operator import itemgetter


class _DictWrapper:
    """An object that contains a dictionary."""

    def __init__(self, d=None, name=''):
        # if d is provided, use it; otherwise make a new dict
        if d == None:
            d = {}
        self.d = d
        self.name = name

    def GetDict(self):
        """Gets the dictionary."""
        return self.d

    def Values(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is the the keys ion this
        dictionaries are the values of the Hist/Pmf, and the
        values are frequencies/probabilities.
        """
        return self.d.keys()

    def Items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def Render(self):
        """Generates a sequence of points suitable for plotting.

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*sorted(self.Items()))

    def Print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.iteritems()):
            print(val, prob)

    def Set(self, x, y=0):
        """Sets the freq/prob associated with  the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def Incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def Total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        """Returns the largest frequency/probability in the map."""
        return max(self.d.itervalues())


class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def Copy(self, name=None):
        """Returns a copy of this Hist.

        Args:
            name: string name for the new Hist
        """
        if name is None:
            name = self.name
        return Hist(dict(self.d), name)

    def Freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def Freqs(self):
        """Gets an unsorted sequence of frequencies."""
        return self.d.values()

    def IsSubset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.Items():
            self.Incr(val, -freq)

    def Mode(self):
        items = [(val, freq) for val, freq in self.Items()]
        return max(items, key=itemgetter(1))[0]

    def AllMode(self):
        items = [(val, freq) for val, freq in self.Items()]
        return [val for val, freq in
                sorted(items, key=itemgetter(1), reverse=True)]


class Pmf(_DictWrapper):
    """Represents a probability mass function.
    
    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def Copy(self, name=None):
        """Returns a copy of this Pmf.

        Args:
            name: string name for the new Pmf
        """
        if name is None:
            name = self.name
        return Pmf(dict(self.d), name)

    def Prob(self, x, default=0):
        """Gets the probability associated with the value x.

        Args:
            x: number value
            default: value to return if the key is not there

        Returns:
            float probability
        """
        return self.d.get(x, default)

    def Probs(self):
        """Gets an unsorted sequence of probabilities."""
        return self.d.values()

    def Normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is 1.

        Args:
            fraction: what the total should be after normalization
        """
        total = self.Total()
        if total == 0.0:
            logging.warning('Normalize: total probability is zero.')
            raise ValueError('total probability is zero.')
        
        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor
    
    def Random(self):
        """Chooses a random element from this PMF.

        Returns:
            float value from the Pmf
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')
            
        target = random.random()
        total = 0.0
        for x, p in self.d.iteritems():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def Mean(self):
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        mu = 0.0
        for x, p in self.d.iteritems():
            mu += p * x
        return mu

    def Var(self, mu=None):
        """Computes the variance of a PMF.

        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean

        Returns:
            float variance
        """
        if mu is None:
            mu = self.Mean()
            
        var = 0.0
        for x, p in self.d.iteritems():
            var += p * (x - mu)**2
        return var

    def Log(self):
        """Log transforms the probabilities."""
        m = self.MaxLike()
        for x, p in self.d.iteritems():
            self.Set(x, math.log(p/m))

    def Exp(self):
        """Exponentiates the probabilities."""
        m = self.MaxLike()
        for x, p in self.d.iteritems():
            self.Set(x, math.exp(p-m))


def MakeHistFromList(t, name=''):
    """Makes a histogram from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this histogram

    Returns:
        Hist object
    """
    hist = Hist(name=name)
    [hist.Incr(x) for x in t]
    return hist


def MakeHistFromDict(d, name=''):
    """Makes a histogram from a map from values to frequencies.

    Args:
        d: dictionary that maps values to frequencies
        name: string name for this histogram

    Returns:
        Hist object
    """
    return Hist(d, name)


def MakePmfFromList(t, name=''):
    """Makes a PMF from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this PMF

    Returns:
        Pmf object
    """
    hist = MakeHistFromList(t, name)
    return MakePmfFromHist(hist)


def MakePmfFromDict(d, name=''):
    """Makes a PMF from a map from values to probabilities.

    Args:
        d: dictionary that maps values to probabilities
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def MakePmfFromHist(hist, name=None):
    """Makes a normalized PMF from a Hist object.

    Args:
        hist: Hist object
        name: string name

    Returns:
        Pmf object
    """
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.GetDict())
    pmf = Pmf(d, name)
    pmf.Normalize()
    return pmf


def MakePmfFromCdf(cdf, name=None):
    """Makes a normalized Pmf from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Pmf

    Returns:
        Pmf object
    """
    if name is None:
        name = cdf.name

    pmf = Pmf(name=name)

    prev = 0.0
    for val, prob in cdf.Items():
        pmf.Incr(val, prob-prev)
        prev = prob

    return pmf


def MakeMixture(pmfs, name='mix'):
    """Make a mixture distribution.

    Args:
      pmfs: Pmf that maps from Pmfs to probs.
      name: string name for the new Pmf.

    Returns: Pmf object.
    """
    mix = Pmf(name=name)
    for pmf, prob in pmfs.Items():
        for x, p in pmf.Items():
            mix.Incr(x, p * prob)
    return mix


if __name__ == '__main__':
    data = [1, 2, 2, 3, 3, 3, 5]
    hist = MakeHistFromList(data)
    print('hist object: ', hist)
    print('freq of 2: ', hist.Freq(2))
    print('freq of 3: ',hist.Freq(3))
    print('freq of 4', hist.Freq(4))
    print('list of value: ', hist.Values())

    for val in sorted(hist.Values()):
        print('value: {0}, freq {1}'.format(val, hist.Freq(val)))

    for val, freq in hist.Items():
        print('value: {0}, freq: {1}'.format(val, freq))

    print('Mode: ', hist.Mode())
    print('All Mode: ', hist.AllMode())

    n = hist.Total()
    pmf = {}
    for x, freq in hist.Items():
        pmf[x] = freq / n

    print(pmf)

    pmf2 = MakePmfFromList(data)
    for val, prob in pmf2.Items():
        print('{0}: {1}'.format(val, prob))
