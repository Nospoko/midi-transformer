import json

import fortepyan as ff
from datasets import DatasetInfo

from downstream_task_datasets.SubSequenceDataset import SubSequenceDataset

_DESC = """
Dataset with midi files, with bass extracted to tgt_notes and the rest left in src_notes columns.
Each record has the same sum of notes.
"""


class BassExtractedDataset(SubSequenceDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def create_record(self, piece: ff.MidiPiece):
        notes = piece.df
        bass_ids = notes.pitch < 48
        src_notes = notes[~bass_ids]
        tgt_notes = notes[bass_ids]

        record = {
            "src_notes": src_notes,
            "tgt_notes": tgt_notes,
            "extracted": ["bass"],
            "source": json.dumps(piece.source),
        }
        return record
