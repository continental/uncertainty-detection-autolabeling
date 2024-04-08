The available config files are for the two datasets KITTI and BDD. Replace content as necessary. This config is for the file `src/train_flags.py`.

For `BDD`: All configs ending with `BDD`.

For `KITTI`: All other configs.

The following training options are availble:
 - `orig`: Model without uncertainty.
 - `lossatt`: Model with Loss Attenuation.
 - `mcdropout`: Model with full MC-Dropout.
 - `mcdropout_lossatt`: Model with full MC-Dropout + Loss Attenuation.
 - `mcdropout_head`: Model with MC-Dropout in detector head only.
 - `mcdropout_lossatt_head`: Model with MC-Dropout in detector head only + Loss Attenuation.

`train_runner.ini` allows the pre-definition of the args and running the training command via `src/train_runner.py`. Comment out whichever dataset you want to run and replace content as necessary.