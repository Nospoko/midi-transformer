import json

import fortepyan as ff
from datasets import DatasetInfo

from artifacts import get_voice_range
from midi_datasets.MidiSequenceDataset.MidiSequenceDataset import MidiSequenceDataset

_DESC = """
Dataset with midi files, with bass extracted to target_notes and the rest left in source_notes columns.
Each record has the same sum of notes.
"""


class BassExtractedDataset(MidiSequenceDataset):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(description=_DESC)

    def create_record(self, piece: ff.MidiPiece) -> tuple[dict, bool]:
        notes = piece.df
        bass_range = get_voice_range("bass")
        bass_ids = (notes.pitch < bass_range[1]) & (notes.pitch > bass_range[0])
        source_notes = notes[~bass_ids]
        target_notes = notes[bass_ids]

        record = {
            "source_notes": source_notes,
            "target_notes": target_notes,
            "extraction_type": "bass",
            "source": json.dumps(piece.source),
        }
        return record

    def validate_record(self, record: dict):
        if len(record["source_notes"]) > 0 and len(record["target_notes"]) > 0:
            return True
        return False
