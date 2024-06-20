import json

import fortepyan as ff
from datasets import DatasetInfo

from artifacts import extraction_type_to_range
from downstream_task_datasets.SubSequenceDataset import SubSequenceDataset

_DESC = """
Dataset with midi files, with bass extracted to target_notes and the rest left in source_notes columns.
Each record has the same sum of notes.
"""


class BassExtractedDataset(SubSequenceDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def create_record(self, piece: ff.MidiPiece):
        notes = piece.df
        bass_range = extraction_type_to_range["bass"]
        bass_ids = (notes.pitch < bass_range[1]) & (notes.pitch > bass_range[0])
        source_notes = notes[~bass_ids]
        target_notes = notes[bass_ids]

        record = {
            "source_notes": source_notes,
            "target_notes": target_notes,
            "extracted": ["bass"],
            "source": json.dumps(piece.source),
        }
        return record
