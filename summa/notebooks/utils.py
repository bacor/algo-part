import matplotlib.pyplot as plt
import music21
from music21 import stream
from music21 import instrument
from music21.articulations import BreathMark
from music21.meter import TimeSignature
from music21.expressions import RehearsalMark

plt.rcParams['lines.solid_capstyle'] = 'round'

def despine(ax=None):
    if ax is None: ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

def cm2inch(*args):
    return list(map(lambda x: x/2.54, args))

def title(title, ax=None, x=0, ha='left', fontweight='bold', **kwargs):
    if ax is None: ax = plt.gca()
    ax.set_title(title, x=x, ha=ha, fontweight=fontweight, **kwargs)
    
def tintinnabuli_grid(ax=None, grid=True):
    if ax is None: ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    labels = [f'E{octave}' for octave in [2, 3, 4, 5]]
    ticks = [pitch_to_note[label] for label in labels]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    if grid:
        ax.yaxis.set_tick_params(length=0)
        ax.grid(axis='y', c='0.85', zorder=1, dashes=(5, 10)) 
    ax.set_ylim(y_min, y_max)
    
def num_measures(score):
    """Computes the total number of measures in a score"""
    return len(score.parts[0].recurse(classFilter='Measure'))

def insert_breathmark(note):
    """Insert a breathing mark after a note"""
    breath = BreathMark()
    breath.symbol = 'comma'
    breath.placement = 'above'
    note.articulations.append(breath)
    
def add_rehearsal_marks(score, period=3):
    """Adds a rehearsal mark every other three measures"""
    for m in range(num_measures(score)):
        if m % period == 0:
            mark = RehearsalMark(int(m / period) + 1)
            mark.placement = 'above'
            measure = score.parts[0].measure(m+1)
            measure.insert(0, mark)
            
def set_time_signatures(score, hidden=True):
    """Update the time signature of every measure to match the
    number of quarter notes in that measure"""
    for m in range(num_measures(score)):
        bar_duration = int(score.measure(m+1).quarterLength)
        if bar_duration == 0: continue
        t = TimeSignature(f'{bar_duration}/4')
        if hidden:
            t.style.hideObjectOnPrint = True
        for part in score.parts:
            part.measure(m+1).timeSignature = t
            
def syllabify(text):
    """Split text in syllables at dashes, in words at spaces
    and phrases at line breaks."""
    syllables = []
    for phrase in text.split('\n'):
        for word in phrase.split(' '):
            sylls = word.split('-')
            for i, syll in enumerate(sylls):
                if i < len(sylls) - 1:
                    syll += '-'
                syllables.append(syll)
    return syllables


def init_SATB_score(key_signature=None, metadata=None):
    """Returns an score object where all parts, clefs, key signatures
    and metadata have been added"""
    # Parts
    part_S = stream.Part(id='soprano')
    part_A = stream.Part(id='alto')
    part_T = stream.Part(id='tenor')
    part_B = stream.Part(id='bass')
    part_S.append(instrument.Soprano())
    part_A.append(instrument.Alto())
    part_T.append(instrument.Tenor())
    part_B.append(instrument.Bass())
    
    # Score
    score = stream.Score()
    score.append([part_S, part_A, part_T, part_B])
    
    # Staff group with bracket
    staff_group = music21.layout.StaffGroup(
        score.parts, symbol='bracket', barTogether=False)
    score.insert(0, staff_group)

    # Insert first measure
    for part in score.parts:
        part.append(stream.Measure())

    # Set clefs
    bass_clef = music21.clef.BassClef()
    part_B.measure(1).insert(0, bass_clef)
    tenor_clef = music21.clef.Treble8vbClef()
    part_T.measure(1).insert(0, tenor_clef)
    
    # Key signature
    if key_signature:
        for part in score.parts:
            part.measure(1).insert(0, key_signature)
    
    # Set metadata
    if metadata:
        score.insert(0, metadata)
    
    return score