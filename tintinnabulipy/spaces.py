# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         spaces.py
# Purpose:      Define the melody and tintinnabuli pitch spaces
#
# Authors:      Bas Cornelissen
#
# Copyright:    Copyright © 2022-present Bas Cornelissen
# License:      see LICENSE
# ------------------------------------------------------------------------------
from __future__ import annotations
from typing import Union, Iterable, Optional

import numpy as np
import music21
from music21.note import Note
from music21.pitch import Pitch
from music21.chord import Chord
from music21.scale import ConcreteScale
from music21.scale.intervalNetwork import Direction


import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator

# Used in doctests
from music21.scale import MajorScale

# To do
# -----
# - The type Pitch should usually be Union[Pitch, str]


class MultipleIndexLocator(IndexLocator):
    def __init__(self, base, offset: Iterable, **kwargs):
        """A locator that repeats multiple, possibly unevenly spaced values.
        For example:

        >>> loc = MultipleIndexLocator(10, [2, 5])
        >>> loc.tick_values(0, 20)
        array([ 2,  5, 12, 15])

        This is used to display the grid of a subspace.

        Parameters
        ----------
        base : float
            The base, basically the period with which the ticks are repeated
        offset : Iterable
            Iterable of offsets, each smaller than the base. In every period all
            these offsets are added as ticks
        """
        return super().__init__(base, offset, **kwargs)

    def tick_values(self, vmin, vmax):
        return self.raise_if_exceeds(
            np.sort(
                np.concatenate(
                    [np.arange(vmin + o, vmax + 1, self._base) for o in self.offset]
                )
            )
        )


class NotInSpaceException(Exception):
    pass


class OutsideSpaceException(Exception):
    pass


class ScalarPitchSpace:
    def __init__(
        self,
        scale: Optional[ConcreteScale] = None,
        pitches: Optional[Iterable[Pitch]] = None,
        center: Optional[Pitch] = None,
        name: Optional[str] = None,
    ) -> ScalarPitchSpace:

        if scale is None and pitches is None:
            raise ValueError("You have to pass either a scale or a list of pitches")
        if pitches is not None:
            scale = ConcreteScale(pitches=pitches, tonic=pitches[0])

        self.scale = scale

        if center is None:
            self.center = self.scale.tonic
        else:
            self.center = Pitch(center)

        self.name = name
        if name is None:
            scale_name = (
                self.scale.name.replace("Concrete", "").replace("Abstract", "").strip()
            )
            if len(scale_name) > 1:
                self.name = self.scale.name

        self.pitches = self.scale.getPitches(
            minPitch=f"{self.center.name}0", maxPitch=f"{self.center.name}8"
        )
        zero = self.pitches.index(self.center)
        self.steps = np.arange(len(self.pitches)) - zero
        self.step_to_pitch = {s: p for s, p in zip(self.steps, self.pitches)}
        self.pitch_to_step = {p: s for s, p in self.step_to_pitch.items()}

    def __repr__(self) -> str:
        name = self.name
        if name is None:
            pitch_classes = [p.name for p in self.scale.pitches[:-1]]
            name = "(" + ", ".join(pitch_classes) + ")"
        return f"<{self.__class__.__name__} {name} center={self.center}>"

    def __eq__(self, other: ScalarPitchSpace) -> bool:
        """Test whether two ScalarPitchSpaces are the same by testing whether
        their scales are the same and they have the same center

        Parameters
        ----------
        other : ScalarPitchSpace
            The other pitch space

        Returns
        -------
        bool
            True if the two pitch spaces have the same scale and center
        """
        return (self.scale == other.scale) and (self.center == other.center)

    def __contains__(self, pitch: Union[Pitch, str]) -> bool:
        """Test if the given pitch is in this space

        Parameters
        ----------
        pitch : Union[Pitch, str]
            The pitch, either as a pitch object or as a string (e.g. E4)

        Returns
        -------
        bool
            True if the pitch is in the space
        """
        return Pitch(pitch) in self.pitches

    def is_subspace(self, subspace: ScalarPitchSpace) -> bool:
        """Test if another space is a subspace of this space by testing if all
        its pitches are in this space.

        Parameters
        ----------
        subspace : ScalarPitchSpace
            The other space which might be a subspace

        Returns
        -------
        bool
            True if all pitches from subspace are in this space
        """
        for pitch in subspace.scale.pitches:
            if pitch not in self:
                return False
        return True

    def pitch(self, step: Union[int, Iterable[Union[Pitch, str]]]) -> Pitch:
        if np.iterable(step):
            return [self.pitch(s) for s in step]
        return self.step_to_pitch[step]

    def step(self, pitch: Union[Pitch, str, Iterable[Union[Pitch, str]]]):
        if not type(pitch) is str and np.iterable(pitch):
            return [self.step(p) for p in pitch]
        return self.pitch_to_step[Pitch(pitch)]

    def move(self, origin: Pitch, distance: int) -> Pitch:
        """Move to the pitch in the space that is the specified distance away

        >>> S = ScalarPitchSpace(MajorScale('C3'))
        >>> S.move('C3', 1)
        <music21.pitch.Pitch D3>
        >>> S.move('C3', -3)
        <music21.pitch.Pitch G2>

        Parameters
        ----------
        origin : Pitch
            The starting pitch
        steps : int
            The distance to the target pitch

        Returns
        -------
        Pitch
            The target pitch that is the specified distance from the origin

        Raises
        ------
        NotInSpaceException
            Raised when the origin is not in the space
        """
        if origin not in self:
            raise NotInSpaceException("The pitch is not in this space")

        # To do: which one is faster? using indexing or using music21?

        # if distance == 0:
        #     return Pitch(origin)
        # return self.scale.next(
        #     pitchOrigin=origin,
        #     direction='ascending' if distance >=0 else 'descending',
        #     stepSize=np.abs(distance),
        #     getNeighbor=False)

        index = self.pitch_to_step[Pitch(origin)]
        if index + distance not in self.step_to_pitch:
            raise OutsideSpaceException("Distance is too large")
        return self.step_to_pitch[index + distance]

    def up(self, origin: Pitch) -> Pitch:
        """Move one step up from the origin.

        This is a shorthand for the .move method.

        Parameters
        ----------
        origin : Pitch
            The starting pitch

        Returns
        -------
        Pitch
            The first pitch in the space above the origin
        """
        return self.move(origin, 1)

    def down(self, origin: Pitch) -> Pitch:
        """Move one step down from the origin.

        This is a shorthand for the .move method.

        Parameters
        ----------
        origin : Pitch
            The starting pitch

        Returns
        -------
        Pitch
            The first pitch in the space below the origin
        """
        return self.move(origin, -1)

    def sequence(self, origin: Pitch, steps: int) -> Iterable[Pitch]:
        """Return a sequence of consecutive pitches in the space. The sequence
        always starts with the origin. A sequence of two steps therefore
        contains three notes: the origin, and the next two steps.

        >>> S = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'])
        >>> S.sequence('C4', 0)
        [<music21.pitch.Pitch C4>]
        >>> S.sequence('C4', -2)
        [<music21.pitch.Pitch C4>, <music21.pitch.Pitch G3>, <music21.pitch.Pitch E3>]

        Parameters
        ----------
        origin : Pitch
            The starting pitch
        steps : int
            The number of steps to take. When steps is positive, the sequence
            moves up, when it is negative the sequence moves down.

        Returns
        -------
        Iterable[Pitch]
            A sequence of pitches
        """
        if steps == 0:
            return [Pitch(origin)]
        direction = 1 if steps > 0 else -1
        distances = range(0, steps + direction * 1, direction)
        return [self.move(origin, d) for d in distances]

    def neighbor(self, pitch: Pitch, position: int) -> Pitch:
        """Project a pitch onto the scale in a certain position. If p is the
        position, this method returns the step that is |p| scale steps away from
        the pitch. This reformulates the concept of tintinnabuli positions."""
        if position == 0:
            if pitch in self:
                return Pitch(pitch)
            else:
                return None

        return self.scale.next(
            pitch,
            stepSize=np.abs(position),
            direction=Direction.DESCENDING if position < 0 else Direction.ASCENDING,
            getNeighbor=True,
        )

    def neighbor_position(
        self, reference: Pitch, neighbor: Pitch, max_iter: Optional[int] = 100
    ) -> int:
        """Find the position of a neighbor pitch with respect to another pitch.
        That is, the position p such that the neighbor is the p-th neighbor
        of the reference pitch.

        >>> S = ScalarPitchSpace(MajorScale('C3'))
        >>> S.neighbor_position('C#3', 'D3')
        1
        >>> S.neighbor_position('C#3', 'F3')
        3
        >>> S.neighbor_position('C#3', 'B2')
        -2

        Parameters
        ----------
        reference : Pitch
            The reference pitch
        neighbor : Pitch
            The neighbor from this space
        max_iter : Optional[int], optional
            Maximum number if iterations, by default 100

        Returns
        -------
        int
            The position

        Raises
        ------
        Exception
            Raised if the neighbor and pitch are identical or if the number of
            iterations has been exceeded
        NotInSpaceException
            Raised if the neighbor is not in this space
        """
        reference = Pitch(reference)
        neighbor = Pitch(neighbor)
        if reference == neighbor:
            return 0
        if neighbor not in self:
            raise NotInSpaceException("The neighbor is not in this space")

        sign = +1 if neighbor >= reference else -1
        pos = sign
        i = 0
        while self.neighbor(reference, pos) != neighbor and i < max_iter:
            pos += sign
            i += 1
        if i == max_iter:
            raise Exception("Maximum number of iterations exceeded")
        return pos

    def distance(self, pitch_a, pitch_b):
        """Measures the number of scale steps between pitches A and B.
        This works for any pitch."""
        NotImplemented
        # segment = self.scale.getPitches(minPitch=pitch_a, maxPitch=pitch_b)
        # return len(segment) - 1

    ### Operations

    def transpose(
        self, pitch: Union[Pitch, Iterable[Pitch]], distance: int
    ) -> Union[Pitch, Iterable[Pitch]]:
        """Transpose a pitch or a sequence of pitches by a certain number
        of steps along the scale.

        This is essentially a shorthand for the move method, that also accepts
        sequences of pitches.

        Parameters
        ----------
        pitch : Union[Pitch, Iterable[Pitch]]
            A pitch or sequence of pitches
        distance : int
            The number of steps to tranpose the pitches by

        Returns
        -------
        Union[Pitch, Iterable[Pitch]]
            A pitch or sequence of tranposed pitches
        """
        if np.iterable(pitch) and not type(pitch) is str:
            return [self.transpose(p, distance) for p in pitch]
        return self.move(pitch, distance)

    def mirror(
        self, pitch: Union[Pitch, Iterable[Pitch]], center: Pitch = None
    ) -> Union[Pitch, Iterable[Pitch]]:
        """Mirror a pitch (sequence) in a certain pitch center (by default the
        center of the space). So if a pitch is d steps away from the center,
        it's mirror image is -d steps away.

        Parameters
        ----------
        pitch : Union[Pitch, Iterable[Pitch]]
            The pitch (sequence)
        center : Pitch, optional
            The center of the mirror, by default the center of the space

        Returns
        -------
        Union[Pitch, Iterable[Pitch]]
            The mirrorered pitch (sequence)
        """
        if center is None:
            center = self.center
        if np.iterable(pitch) and not type(pitch) is str:
            return [self.mirror(p, center=center) for p in pitch]
        return self.pitch(2 * self.step(center) - self.step(pitch))

    ### Plotting

    def plot(self, melody, *args, xs=None, ax=None, **kwargs):
        """"""
        if ax is None:
            ax = plt.gca()
        if xs is None:
            xs = np.arange(len(melody))
        return ax.plot(xs, self.step(melody), *args, **kwargs)

    def set_major_grid(self, locator, label=True, ax=None, **kwargs):
        """"""
        if ax is None:
            ax = plt.gca()
        ax.yaxis.set_major_locator(locator)
        if label:
            formatter = lambda x, _: self.pitch(x).nameWithOctave
            ax.yaxis.set_major_formatter(formatter)
        ax.grid(which="major", axis="y", **kwargs)

    def set_minor_grid(self, locator, label=False, ax=None, c="0.85", lw=.5,
    dashes=(3, 5), **kwargs):
        """"""
        if ax is None:
            ax = plt.gca()
        ax.yaxis.set_minor_locator(locator)
        if label:
            formatter = lambda x, _: self.pitch(x).nameWithOctave
            ax.yaxis.set_minor_formatter(formatter)
        ax.grid(which="minor", axis="y", zorder=1, c=c, dashes=dashes, lw=lw, **kwargs)
        ax.tick_params(which="minor", length=0)

    def grid(self, ax=None, hide_xticks=True, minor_kws=dict(), major_kws=dict()):
        """"""
        if ax is None:
            ax = plt.gca()
        N = len(self.scale.pitches) - 1
        self.set_major_grid(MultipleLocator(N), ax=ax, **major_kws)
        self.set_minor_grid(AutoMinorLocator(N), ax=ax, **minor_kws)
        if hide_xticks:
            ax.set_xticks([])

    def subspace_grid(
        self,
        other,
        label_minor: bool = True,
        label_major=True,
        hide_xticks=True,
        ax=None,
    ):
        """"""
        if ax is None:
            ax = plt.gca()
        if not self.is_subspace(other):
            raise Exception("Foreign space has to be a subspace")
        N = len(self.scale.pitches) - 1

        # Major ticks are octaves: evenly spaced, but possibly shifted
        zero = self.step(other.center)
        self.set_major_grid(IndexLocator(N, zero), label=label_major, ax=ax)

        # Minor ticks from another space are possibly unevenly spaced
        offsets = [self.step(p) for p in other.scale.pitches[1:-1]]
        self.set_minor_grid(MultipleIndexLocator(N, offsets), label=label_minor, ax=ax)
        if hide_xticks:
            ax.set_xticks([])


class MelodicSpace(ScalarPitchSpace):
    def mode1(self, length: int, center: Optional[Pitch] = None) -> Iterable[Pitch]:
        """Return a mode 1 melody that moves up from the center

        >>> M = MelodicSpace(MajorScale('C3'))
        >>> M.mode1(2)
        [<music21.pitch.Pitch C3>, <music21.pitch.Pitch D3>, <music21.pitch.Pitch E3>]

        Parameters
        ----------
        steps : int
            The number of steps
        center : Optional[Pitch], optional
            The pitch center, by default the center of the space

        Returns
        -------
        Iterable[Pitch]
            A sequence of pitches moving up from the center

        Raises
        ------
        ValueError
            Raised if the length is negative
        """
        if center is None:
            center = self.center
        if length < 0:
            raise ValueError("The length has to be >= 0")
        return self.sequence(center, length)

    def mode2(self, length: int, center: Optional[Pitch] = None) -> Iterable[Pitch]:
        """Return a mode 2  melody that moves down from the center

        >>> from music21.scale import MajorScale
        >>> M = MelodicSpace(MajorScale('C3'))
        >>> M.mode2(2)
        [<music21.pitch.Pitch C3>, <music21.pitch.Pitch B2>, <music21.pitch.Pitch A2>]

        Parameters
        ----------
        length : int
            The length of the melody
        center : Optional[Pitch], optional
            The pitch center, by default the center of the space

        Returns
        -------
        Iterable[Pitch]
            A mode 2 melody that descends down from the center

        Raises
        ------
        ValueError
            Raised if the length is negative
        """
        if center is None:
            center = self.center
        if length < 0:
            raise ValueError("The length has to be >= 0")
        return self.sequence(center, -1 * length)

    def mode3(self, steps: int, center: Optional[Pitch] = None) -> Iterable[Pitch]:
        """Move down towards the pitch center"""
        return retrograde(self.mode1(steps, center=center))

    def mode4(self, steps: int, center: Optional[Pitch] = None) -> Iterable[Pitch]:
        """Move up towards the pitch center"""
        return retrograde(self.mode2(steps, center=center))


class TintinnabuliSpace(ScalarPitchSpace):
    """A tintinnabuli space.

    Parameters
    ----------
    chord : Optional[Chord]
        The chord used to build the space
    pitches : Optional[Iterable[Union[Pitch, str]]], optional
        Alternatively, a set of pitches, by default None
    name : Optional[str], optional
        Name of the chord or space, by default None
    """

    def __init__(
        self,
        chord: Optional[Chord] = None,
        pitches: Optional[Iterable[Union[Pitch, str]]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> TintinnabuliSpace:
        """"""
        if pitches is not None:
            chord = Chord(pitches)
        if name is None:
            name = chord.pitchedCommonName
        self.chord = chord
        super().__init__(pitches=chord.pitches, name=name, **kwargs)


### Operations


def concatenate(*args: Iterable) -> Iterable:
    return [el for arg in args for el in arg if len(arg) > 0]


def glue(*args: Iterable) -> Iterable:
    """Glue multiple sequences together. If two consecutive sequences start
    with the same element, one is removed to avoid a repetition.

    >>> glue([1, 2], [2, 3], [4])
    [1, 2, 3, 4]
    >>> glue([1, 2], [2, 3], [3])
    [1, 2, 3]

    Returns
    -------
    Iterable
        The sequence obtained by glueing all arguments together
    """
    args = [a if np.iterable(a) else [a] for a in args]
    for i in range(0, len(args) - 1):
        if args[i][-1] == args[i + 1][0]:
            args[i] = args[i][:-1]
    return concatenate(*args)


def retrograde(sequence: Iterable) -> Iterable:
    return sequence[::-1]


def rotate(sequence: Iterable, distance: int) -> Iterable:
    """Cyclically rotate a sequence to the left

    >>> rotate([1, 2, 3, 4], 2)
    [3, 4, 1, 2]
    >>> rotate([1, 2, 3, 4], -1)
    [4, 1, 2, 3]

    Parameters
    ----------
    sequence : Iterable
        The sequence
    distance : int
        The number of steps to rotate

    Returns
    -------
    Iterable
        The rotated sequence
    """
    N = len(sequence)
    return [sequence[(i + distance) % N] for i in range(N)]


def rotate_tail(
    sequence: Iterable, distance: int, start: Optional[int] = 1
) -> Iterable:
    """Rotate the final part of a sequence, leaving the head of the sequence
    untouched.

    >>> seq = [1, 2, 3, 4, 5]
    >>> rotate_tail(seq, distance=1)
    [1, 3, 4, 5, 2]
    >>> rotate_tail(seq, distance=-2)
    [1, 4, 5, 2, 3]
    >>> rotate_tail(seq, distance=1, start=2)
    [1, 2, 4, 5, 3]

    Parameters
    ----------
    sequence : Iterable
        The sequence
    distance : int
        The number of steps to rotate by
    start : int
        Index of the start of the tail, by default 1

    Returns
    -------
    Iterable
        The partially rotated sequence
    """
    head = list(range(0, start))
    tail = rotate(range(start, len(sequence)), distance)
    return [sequence[i] for i in head + tail]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
