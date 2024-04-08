The available config files are for the three datasets KITTI, BDD and CODA. Replace content as necessary. This config is for the file `src/eval.py`.

For `KITTI`:
 - `eval_k`: Standard evaluation config file.
 - `eval_kc`: Config file to eval KITTI models on CODA dataset.
 - `eval_ks`: Config file to eval KITTI models on split KITTI val set used in `src/active_learning_eval.py`.

For `BDD`:
 - `eval_b`: Standard evaluation config file.
 - `eval_bc`: Config file to eval BDD models on CODA dataset.
 - `eval_bs`: Config file to eval BDD models on split BDD val set used in `src/active_learning_eval.py`.

For `CODA`:
 - `eval_cbs`: onfig file to eval KITTI models on split CODA test set used in `src/active_learning_eval.py`.
 - `eval_cks`: Config file to eval KITTI models on split CODA test set used in `src/active_learning_eval.py`.