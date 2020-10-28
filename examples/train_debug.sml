local train_tmpl = import "train_tmpl.libsm";
local cellsize_lr_tune_job(cell_size, lr) = train_tmpl {
    vars: {
        train_base: "/tmp/kinase/",
        data_path: "/home/zhengxu/data/kinase/"
    },
    dataset_spec: {
        train_csv_path: "%s%s" % [$.vars.data_path, "train.smi"],
        val_csv_path: "%s%s" % [$.vars.data_path, "val.smi"],
        test_csv_path: "%s%s" % [$.vars.data_path, "test.smi"]
    },
    data_hparams: {
        reader: "DiscoveryReader",
        # buckets: [50, 100],
        val_batch_size: 100
    },
    hparams: {
        cell_size: cell_size,
        init_learning_rate: lr,
        embed_dim: 0
    },
    args: super.args {
        epochs: 200,
        steps_per_checkpoint: 100
    },
    name: "gen-gru-%d-lr-%s" % [
        self.hparams.cell_size,
        std.strReplace(std.toString(self.hparams.init_learning_rate), ".", "-dot-")]
};
cellsize_lr_tune_job(128, 5e-3)
