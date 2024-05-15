```
                  _   ___              ___               _             _       _             
   ___ ___  _ __ | |_|_ _|_ __  _   _ / _ \ _   _ ___   | |_ _ __ __ _(_)_ __ (_)_ __   __ _ 
  / __/ _ \| '_ \| __|| || '_ \| | | | | | | | | / __|  | __| '__/ _` | | '_ \| | '_ \ / _` |
 | (_| (_) | | | | |_ | || | | | |_| | |_| | |_| \__ \  | |_| | | (_| | | | | | | | | | (_| |
  \___\___/|_| |_|\__|___|_| |_|\__,_|\___/ \__,_|___/___\__|_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                    |_____|                            |___/ 
```

cd design/module/fine-grained

python3 main.py -path /mnt/extra/continuous-training/dataset/test_8_hours -dataset_name profile_v1.feat_v6_ts.readonly.dataset -data_train_duration_min 15 -data_retrain_duration_min 1 -data_eval_duration_min 1 -model_algo nn_clf -model_name nn_clf -dd_algo ip-based -output nn_clf_ip_based