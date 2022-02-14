import unittest
from tintanibullipy import *
from music21.scale import MajorScale
from music21.scale import MinorScale
from music21.scale import ConcreteScale
from music21.pitch import Pitch
from music21.chord import Chord

Cmajor = ScalarPitchSpace(MajorScale('C4'))
Ctriad = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'])

class TestScalarPitchSpace(unittest.TestCase):

    def test_initialize(self):
        Cmajor = ScalarPitchSpace(MajorScale('C4'))
        self.assertEqual(Cmajor.center, Pitch('C4'))

        Ctriad = ScalarPitchSpace(ConcreteScale(pitches=['C4', 'E4', 'G4']))
        self.assertEqual(Ctriad.center, Pitch('C4'))

        Ctriad2 = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'])
        self.assertEqual(Ctriad2.center, Pitch('C4'))

    def test_repr(self):
        self.assertEqual(str(Cmajor), '<ScalarPitchSpace C major center=C4>')
        self.assertEqual(str(Ctriad), '<ScalarPitchSpace (C, E, G) center=C4>')

    def test_equality(self):
        Ctriad = ScalarPitchSpace(ConcreteScale(pitches=['C4', 'E4', 'G4']))
        Ctriad2 = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'])
        self.assertEqual(Ctriad, Ctriad2)
        Ctriad3 = ScalarPitchSpace(pitches=['C4', 'E4', 'G4'], center='E4')
        self.assertNotEqual(Ctriad, Ctriad3)

    def test_contains(self):
        self.assertTrue(Pitch('C4') in Cmajor)
        self.assertTrue('C2' in Cmajor)
        self.assertFalse('C#' in Cmajor)

    def test_step(self):
        self.assertEqual(Cmajor.step(Pitch('C4')), 0)
        self.assertEqual(Cmajor.step('C4'), 0)

    def test_step_iterable(self):
        pitches = [Pitch('C4'), Pitch('D4'), Pitch('E4')]
        self.assertListEqual(Cmajor.step(pitches), [0, 1, 2])
        self.assertListEqual(Cmajor.step(['C4', 'D4', 'E4']), [0, 1, 2])
    
    def test_pitch(self):
        self.assertEqual(Cmajor.pitch(0), Pitch('C4'))

    def test_pitch_iterable(self):
        self.assertListEqual(Cmajor.pitch([0, 1]), [Pitch('C4'), Pitch('D4')])

    def test_subspace(self):
        self.assertTrue(Cmajor.is_subspace(Ctriad))
        self.assertTrue(Cmajor.is_subspace(Cmajor))
        self.assertFalse(Ctriad.is_subspace(Cmajor))

    def test_move(self):
        self.assertEqual(Cmajor.move('C4', 1), Pitch('D4'))
        self.assertEqual(Cmajor.move('D4', 1), Pitch('E4'))
        self.assertEqual(Cmajor.move('E4', 1), Pitch('F4'))
        self.assertEqual(Cmajor.move('B4', 1), Pitch('C5'))
        self.assertEqual(Cmajor.move('C4', 0), Pitch('C4'))
        self.assertEqual(Cmajor.move('C4', -1), Pitch('B3'))
        self.assertEqual(Cmajor.move('C4', -2), Pitch('A3'))

        self.assertRaises(NotInSpaceException, lambda: Cmajor.move('C#3', 1))
    
    def test_sequence(self):
        self.assertListEqual(Cmajor.sequence('C4', 0), [Pitch('C4')])
        self.assertListEqual(Cmajor.sequence('C4', 1), [Pitch('C4'), Pitch('D4')])
        self.assertListEqual(Cmajor.sequence('C4', -1), [Pitch('C4'), Pitch('B3')])

    def test_neighbor_position(self):
        self.assertEqual(Ctriad.neighbor_position('D4', 'E4'), 1)
        self.assertEqual(Ctriad.neighbor_position('D4', 'G4'), 2)
        self.assertEqual(Ctriad.neighbor_position('D4', 'C4'), -1)
        # self.assertRaises(Exception, lambda: Ctriad.neighbor_position('C4', 'C4'))


class TestMelodicPitchSpace(unittest.TestCase):

    def test_repr(self):
        M = MelodicSpace(MajorScale('C4'))
        self.assertEqual(str(M), '<MelodicSpace C major center=C4>')

    def test_modes(self):
        M = MelodicSpace(MajorScale('C3')) 
        self.assertListEqual(M.mode1(2), [Pitch('C3'), Pitch('D3'), Pitch('E3')])
        self.assertListEqual(M.mode2(2), [Pitch('C3'), Pitch('B2'), Pitch('A2')])
        self.assertListEqual(M.mode3(2), [Pitch('E3'), Pitch('D3'), Pitch('C3')])
        self.assertListEqual(M.mode4(2), [Pitch('A2'), Pitch('B2'), Pitch('C3')])


class TestTintanibulliSpace(unittest.TestCase):

    def test_repr(self):
        T = TintanibulliSpace(Chord(['C4', 'E4', 'G4']))
        self.assertEqual(str(T), '<TintanibulliSpace C-major triad center=C4>')