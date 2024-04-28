from abc import abstractmethod

from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset as TorchDataset
from midi_tokenizers.midi_tokenizer import MidiTokenizer
from augmentation import pitch_shift, change_speed


class MidiDataset(TorchDataset):
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        tokenizer: MidiTokenizer,
        augmentation_probability: float = 0.0,
    ):
        super().__init__()

        # MidiTokenizer which was used during creation of the dataset
        self.tokenizer = tokenizer

        # Dataset with tokenized MIDI data
        self.dataset = dataset
        
        # Probability that any given record should be preprocessed
        self.augmentation_probability = augmentation_probability

    def __len__(self) -> int:
        return len(self.dataset)
    
    def augmentation(self, record: dict):
        notes = self.tokenizer.decode(record["note_token_ids"])
        pitch = notes["pitch"]
        pitch_shifted = pitch_shift(pitch=pitch)
        notes["pitch"] = pitch_shifted
         
        next_start = notes["start"].shift(-1)
        dstart = next_start - notes["start"] 
        
        dstart_changed, duration_changed = change_speed(
            dstart=dstart, 
            duration=notes["duration"],
        )
        
        notes["start"] = dstart_changed.cumsum().shift(1).fillna(0)
        notes["end"] = notes["start"] + duration_changed
        
        note_token_ids = self.tokenizer.encode(notes=notes)
        return {"note_token_ids": note_token_ids, "source": record["source"]}
        
       
    def get_complete_record(self, idx: int) -> dict:
        # The usual token ids + everything we store
        out = self[idx] | self.dataset[idx]
        return out

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass
