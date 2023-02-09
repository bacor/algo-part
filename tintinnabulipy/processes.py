# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         spaces.py
# Purpose:      Implement tintinnabuli processes
#
# Authors:      Bas Cornelissen
#
# Copyright:    Copyright © 2022-present Bas Cornelissen
# License:      see LICENSE
# ------------------------------------------------------------------------------
from __future__ import annotations
from typing import Union, Iterable, Optional

from music21.pitch import Pitch

from .spaces import TintinnabuliSpace
from .spaces import MelodicSpace
from .spaces import rotate_tail

# Doctests
from music21.chord import Chord
from music21.scale import MajorScale


class TintinnabuliProcess:
    """An abstract tintinnabuli process class"""

    def __init__(self, T: TintinnabuliSpace) -> TintinnabuliProcess:
        """An abstract tintinnabuli class

        Parameters
        ----------
        T : TintinnabuliSpace
            The tintinnabuli space
        """
        self.T = T

    def step(self, index: int, melody: Iterable[Pitch], ts: Iterable[Pitch]) -> Pitch:
        """Compute the next pitch in the tintinnabuli process based on melody
        and previous tintinnabuli pitches. The method is overriden by inheriting
        classes.

        Parameters
        ----------
        index : int
            The current index, or position in the melody. The current melody
            pitch can thus be accessed as `melody[index]`
        melody : Iterable[Pitch]
            The melody as a sequence of pitches
        ts : Iterable[Pitch]
            A sequence of previous tintinnabuli pitches of length `index`.
            These previous pitches have also been computed by the step method.

        Returns
        -------
        Pitch
            The next tintinnabuli pitch
        """
        raise NotImplemented

    def __call__(
        self,
        melody: Iterable[Pitch],
        t0: Optional[Pitch] = None,
        p0: Optional[int] = None,
    ) -> Iterable[Pitch]:
        """Compute the tintinnabuli voice for a given melody. You can specify
        the starting pitch as a pitch t0 or as the tintinnabuli position p0.

        Parameters
        ----------
        melody : Iterable[Pitch]
            The melody for which to compute the tintinnabuli process
        t0 : Optional[Pitch], optional
            The initial tintinnabuli pitch, by default None
        p0 : Optional[Pitch], optional
            The initial tintinnabuli position, by default None

        Returns
        -------
        Iterable[Pitch]
            A sequence of tintinnabuli pitches

        Raises
        ------
        Exception
            Raised when no starting pitch for the process is given (either
            as a pitch t0 or as a position p0)
        """
        if p0 is not None:
            t0 = self.T.neighbor(melody[0], p0)
        if t0 is None:
            raise Exception("No starting pitch t0 was given")

        ts = [t0]
        for i in range(1, len(melody)):
            t_cur = self.step(i, melody, ts)
            ts.append(t_cur)
        return ts


class ConstantProcess(TintinnabuliProcess):
    """A constant tintinnabuli process that consistently computes the
    pitch in a constant position.

    >>> M = MelodicSpace(MajorScale('C4'))
    >>> T = TintinnabuliSpace(Chord(['C4', 'E4', 'G4']))
    >>> melody = M.mode1(4)
    >>> [m.nameWithOctave for m in melody]
    ['C4', 'D4', 'E4', 'F4', 'G4']
    >>> tin = ConstantProcess(T, position=1)(melody)
    >>> [t.nameWithOctave for t in tin]
    ['E4', 'E4', 'G4', 'G4', 'C5']
    >>> tin = ConstantProcess(T, position=2)(melody)
    >>> [t.nameWithOctave for t in tin]
    ['G4', 'G4', 'C5', 'C5', 'E5']
    >>> tin = ConstantProcess(T, position=-3)(melody)
    >>> [t.nameWithOctave for t in tin]
    ['C3', 'E3', 'E3', 'G3', 'G3']

    Attributes
    ----------
    position : int
        The position
    """

    def __init__(self, T: TintinnabuliSpace, position: int) -> ConstantProcess:
        """"""
        self.position = position
        super().__init__(T=T)

    def __call__(self, melody: Iterable[Pitch]) -> Iterable[Pitch]:
        return super().__call__(melody, p0=self.position)

    def step(self, index: int, melody: Iterable[Pitch], ts: Iterable[Pitch]) -> Pitch:
        return self.T.neighbor(melody[index], self.position)


class AlternatingProcess(TintinnabuliProcess):
    """Tintinnabuli process that alternates the tintinnabuli position above
    and below the melody.

    Differently put, this process in every step flips the sign of the position.
    The position `p0` of the first tintinnabuli note therefore determines the
    remainder of the process.

    >>> M = MelodicSpace(MajorScale('C4'))
    >>> T = TintinnabuliSpace(Chord(['C4', 'E4', 'G4']))
    >>> melody = M.mode1(4)
    >>> [m.nameWithOctave for m in melody]
    ['C4', 'D4', 'E4', 'F4', 'G4']
    >>> tin = AlternatingProcess(T)(melody, p0=1)
    >>> [t.nameWithOctave for t in tin]
    ['E4', 'C4', 'G4', 'E4', 'C5']
    >>> tin = AlternatingProcess(T)(melody, p0=-3)
    >>> [t.nameWithOctave for t in tin]
    ['C3', 'C5', 'E3', 'E5', 'G3']

    Parameters
    ----------
    TintinnabuliProcess : [type]
        [description]
    """

    def step(self, i: int, melody: Iterable[Pitch], ts: Iterable[Pitch]) -> Pitch:
        prev_pos = self.T.neighbor_position(melody[i - 1], ts[i - 1])
        return self.T.neighbor(melody[i], -1 * prev_pos)


class StepProcess(TintinnabuliProcess):
    def __init__(self, T: TintinnabuliSpace, position: int) -> StepProcess:
        self.position = position
        super().__init__(T=T)

    # TODO: You don't need m_prev

    def _step_positive(self, m_cur, m_prev, t_prev):
        inferior = self.T.down(t_prev)
        minimum = self.T.neighbor(m_cur, self.position)
        if inferior >= minimum:
            return inferior
        else:
            return self.T.neighbor(t_prev, +1)

    def _step_negative(self, m_cur, m_prev, t_prev):
        superior = self.T.up(t_prev)
        maximum = self.T.neighbor(m_cur, self.position)
        if superior <= maximum:
            return superior
        else:
            return self.T.neighbor(t_prev, -1)

    def step(self, i: int, melody: Iterable[Pitch], ts: Iterable[Pitch]) -> Pitch:
        if self.position >= 0:
            return self._step_positive(melody[i], melody[i - 1], ts[i - 1])
        else:
            return self._step_negative(melody[i], melody[i - 1], ts[i - 1])

    def __call__(self, melody: Iterable[Pitch], p0=None, t0=None) -> Iterable[Pitch]:
        if p0 is None and t0 is None:
            p0 = self.position
        return super().__call__(melody, p0=p0, t0=t0)


class OrnamentProcess(TintinnabuliProcess):
    def __init__(self, T, min_pitch, max_pitch, min_target='C0', max_target='C10'):
        self.min_pitch = Pitch(min_pitch)
        self.max_pitch = Pitch(max_pitch)
        self.max_target = Pitch(max_target)
        self.min_target = Pitch(min_target)
        super().__init__(T)
        
    def step(self, i, melody, ts):
        # Very last step
        if i == len(melody) - 1:
            return False
                                
        in_range = self.min_pitch <= melody[i-1] <= self.max_pitch
        target_in_range = self.min_target <= melody[i+1] <= self.max_target
        if (melody[i-1] != melody[i+1]) and in_range and target_in_range:
            return melody[i-1]
        else:
            return False


class TailRotatedPatternProcess(TintinnabuliProcess):
    def __init__(self, T, pattern: Iterable[Pitch]):
        self.pattern = [Pitch(p) if type(p) is str else False for p in pattern]
    
    def step(self, i, melody, ts):
        N = len(self.pattern)
        return rotate_tail(self.pattern, i // N)[i % N]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
