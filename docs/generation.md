## Generation
In gpt2/generation.py there are methods for generating sequences of notes.
For different tasks different generation strategies are used.

#### next_token_prediction task
For this task continuation of prompt notes is being generated.
Model generates max_new_tokens tokens and returns the generated notes.

#### bass_prediction, reverse_bass_prediction and other subsequence tasks

For these tasks other parameters have to be specified:<br>
`whole_prompt_duration` is the duration of fragment for which we want to generate bass.<br>
`prompt_duration` is the duration of sequence that will be input to the model<br>
`target_duration` is the duration of subsequence that will be input to the model <br>
`time_step` is the time step of generation window <br>
The tokens will have a schema: <br>
```
[source_special_token (eg. <NO_BASS>)] [prompt_tokens] [target_special_token] [target_tokens (of target_duration)]
```
In each generation step the target notes will be generated for at most prompt_duration seconds,
and only the notes that ended before time_step + target_duration will be used in the next step

Therefore, prompt_duration should be greater than time_step + target_duration.
