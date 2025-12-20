An experimental UTAU / OpenUtau resampler that improves voicebank realism by correcting timbre using the voicebank’s own recordings. It works as a wrapper around an existing resampler: the normal resampler generates audio first, then a lightweight neural model adjusts the spectral timbre so the result sounds closer to the original samples. One model is trained per voicebank, multipitch banks are supported automatically, and the tool runs like a standard resampler in OpenUtau with a controllable strength setting. The goal is to keep the usual UTAU workflow while making voices sound more natural and consistent, not to replace resamplers, oto.ini, or the UTAU/OpenUtau engine.
Requirements:
- Python 3.9+
- numpy, torch
- A working UTAU/OpenUtau resampler (e.g. doppeltler)
- OpenUtau (required)
Hear it in action.


https://github.com/user-attachments/assets/ddf8bee2-8663-4df0-b6bc-0c4585dcfd0f

