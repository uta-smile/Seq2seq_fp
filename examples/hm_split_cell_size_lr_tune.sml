local train_tmpl = import "train_tmpl.libsm";
local cellsize_lr_tune_job(cell_size, lr) = train_tmpl {
    vars: {
        train_base: "/smile/gfs-nb/home/zhengxu/jobs/drug_gen/hm_lr_cellsize_epoch50/",
        data_path: "/smile/nfs/projects/nih_drug/data/generative/hm_split/"
    },
    dataset_spec: {
        train_csv_path: "%straining_processed_nonan.csv" % $.vars.data_path,
        val_csv_path: "%stest_processed_nonan.csv" % $.vars.data_path
    },
    data_hparams: {
        reader: "GenerativeHMSplitReader",
        val_batch_size: 1000,
        val_data_num: 5000
    },
    hparams: {
        cell_size: cell_size,
        init_learning_rate: lr
    },
    args: super.args {
        epochs: 50
    },
    name: "hm-gen-gru-%d-lr-%s" % [
        self.hparams.cell_size,
        std.strReplace(std.toString(self.hparams.init_learning_rate), ".", "-dot-")]
};
[cellsize_lr_tune_job(cs, lr)
 for cs in [256, 512, 1024]
 for lr in [1e-2, 1e-3]
]
