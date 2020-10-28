local train_tmpl = import "train_tmpl.libsm";
local cellsize_lr_tune_job(cell_size, lr) = train_tmpl {
    vars: {
        train_base: "/smile/gfs-nb/home/zhengxu/jobs/drug_gen/lr_cellsize_epoch50/",
        data_path: "/smile/nfs/projects/nih_drug/data/generative/processed_smi_nonan_preshuffle.atp"
    },
    dataset_spec: {
        csv_path: $.vars.data_path
    },
    data_hparams: {
        val_batch_size: 1000
    },
    hparams: {
        cell_size: cell_size,
        init_learning_rate: lr
    },
    args: super.args {
        epochs: 50
    },
    name: "ep50-reshuffle-gen-gru-%d-lr-%s" % [
        self.hparams.cell_size,
        std.strReplace(std.toString(self.hparams.init_learning_rate), ".", "-dot-")]
};
[cellsize_lr_tune_job(cs, lr)
 for cs in [256, 512, 1024]
 for lr in [1e-2, 1e-3]
]
