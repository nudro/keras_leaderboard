"""
Keras Leaderboard
----------------

A framework for tracking, comparing, and visualizing Keras model performance and metadata.

Example:
    >>> from keras_leaderboard import KerasLeaderboard
    >>> leaderboard = KerasLeaderboard()
    >>> # add models, train, etc.
    >>> leaderboard.display()
"""

from keras_leaderboard.leaderboard import KerasLeaderboard

__version__ = '0.1.0'
__all__ = ['KerasLeaderboard'] 