import unittest
from tintinnabulipy import *
from music21.scale import MajorScale
from music21.scale import MinorScale
from music21.scale import ConcreteScale
from music21.pitch import Pitch
from music21.chord import Chord

Cmajor = ScalarPitchSpace(MajorScale('C4'))
Ctriad = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'])

class TestProcesses(unittest.TestCase):

    def test_constant(self):
        T = TintinnabuliSpace(Chord(['C4', 'E4', 'G4']))
        M = MelodicSpace(MajorScale('C4'))

        melody = M.mode1(4)
        tin = ConstantProcess(T=T, position = 1)(melody)
        target = [Pitch(p) for p in ['E4', 'E4', 'G4', 'G4', 'C5']]
        self.assertSequenceEqual(tin, target)

        tin = ConstantProcess(T=T, position = 2)(melody)
        target = [Pitch(p) for p in ['G4', 'G4', 'C5', 'C5', 'E5']]
        self.assertSequenceEqual(tin, target)

    def test_alternating(self):
        T = TintinnabuliSpace(Chord(['C4', 'E4', 'G4']))
        M = MelodicSpace(MajorScale('C4'))

        melody = [Pitch(p) for p in ['C4', 'D4', 'E4', 'F4', 'G4']]
        target1 = [Pitch(p) for p in ['E4', 'C4', 'G4', 'E4', 'C5']]
        tin1 = AlternatingProcess(T=T)(melody, p0=1)
        self.assertSequenceEqual(tin1, target1)
        
        target2 = [Pitch(p) for p in ['G4', 'G3', 'C5', 'C4', 'E5']]
        tin2 = AlternatingProcess(T=T)(melody, p0=2)
        self.assertSequenceEqual(tin2, target2)

    def test_step(self):
        T = TintinnabuliSpace(Chord(['C4', 'E4', 'G4']))
        M = MelodicSpace(MajorScale('C4'))

        melody = [Pitch(p) for p in ['C4', 'D4', 'E4', 'F4', 'G4']]
        target = [Pitch(p) for p in ['E4', 'G4', 'C5', 'G4', 'C5']]
        tin = StepProcess(T=T, position=1)(melody)
        self.assertSequenceEqual(tin, target)

        melody = [Pitch(p) for p in ['C4', 'D4', 'E4', 'F4', 'G4']]
        target = [Pitch(p) for p in ['G4', 'E4', 'G4', 'C5', 'E5']]
        tin = StepProcess(T=T, position=1)(melody, p0=2)
        self.assertSequenceEqual(tin, target)
