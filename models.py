"""Contains basic model."""

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
from itertools import product
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Fingerprints import FingerprintMols


class DiscoveryModel(object):
    """The simplest generative model."""

    def __init__(self, data_hparams, hparams, vocab, scope="discovery"):
        """Initialize a generative model with only one layer of RNN.
        Args:
            data_hparams: data hyperparameters.
            hparams: model hyper-parameters.
            vocab: A Vocabulary object.
            scope: variable scope prefix for the entire model.
        """
        self._data_hparams = data_hparams
        self._hparams = hparams
        self._vocab = vocab
        self._vocab_size = len(vocab)
        self._scope = scope
    
    def build_embed_inputs(self, encoder_inputs, decoder_inputs, reuse=None):
        vocab_size = len(self._vocab)
        embed_dim = self._hparams.embed_dim
        if embed_dim > 0:
            with tf.variable_scope("input_embed", reuse=reuse):
                if encoder_inputs is not None:
                    encoder_inputs = tf.contrib.layers.embed_sequence(
                        encoder_inputs, vocab_size=vocab_size,
                        embed_dim=embed_dim)
            with tf.variable_scope("input_embed", reuse=True):
                if decoder_inputs is not None:
                    decoder_inputs = tf.contrib.layers.embed_sequence(
                        decoder_inputs, vocab_size=vocab_size,
                        embed_dim=embed_dim)
        return encoder_inputs, decoder_inputs
    
    def build_encoder(self, encoder_inputs, reuse=None):
        with tf.variable_scope("encoder", reuse=reuse):
            cell = tf.contrib.rnn.GRUCell(num_units=self._hparams.cell_size)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                cell, encoder_inputs, dtype=tf.float32)
        return encoder_outputs, encoder_final_state
    
    def build_decoder(self, input_state, helper, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):
            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.GRUCell(self._hparams.cell_size),
                self._vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell, helper=helper, initial_state=input_state)
            # Produce outputs.
            outputs, states, seq_lens = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self._data_hparams.max_seq_len)
        return outputs, states, seq_lens

    def build_train_graph(self, data_inputs):
        vocab_size = len(self._vocab)
        # Build a train graph for the seq2seq model.
        encoder_inputs = data_inputs["encoder_inputs"]
        decoder_inputs = data_inputs["decoder_inputs"]
        decoder_lens = data_inputs["decoder_lens"]
        
        # If we do embed or not.
        encoder_inputs, decoder_inputs = self.build_embed_inputs(
            encoder_inputs, decoder_inputs)
        
        _, encoder_final_state = self.build_encoder(encoder_inputs)

        # Train helper.
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_inputs, decoder_lens)
        
        outputs, states, seq_lens = self.build_decoder(
            encoder_final_state, helper)
        
        return outputs, states, seq_lens
    
    def build_seq2state_op(self, seq_input):
        encoder_inputs, _ = self.build_embed_inputs(
            seq_input, None, reuse=True)

        encoder_outputs, encoder_final_state = self.build_encoder(
            encoder_inputs, reuse=True)

        return encoder_outputs, encoder_final_state
    
    def build_state2seq_op(self, input_state):
        # input_state: [batch_size, state_dim]
        # For testing purpose.
        # An ugly workaround to get a constant tensor with dynamic shape 
        # [batch_size].
        start_tokens = tf.reduce_mean(input_state, axis=-1)
        start_tokens = tf.to_int32(
            (start_tokens * self._vocab.GO_ID) / start_tokens)
        if self._hparams.embed_dim > 0:
            # TODO(zhengxu): Fix the hassle here.
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embeddings, start_tokens=start_token,
                end_token=self._vocab.EOS_ID)
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                lambda x: tf.one_hot(x, self._vocab_size, dtype=tf.float32),
                start_tokens=start_tokens,
                end_token=self._vocab.EOS_ID)
        
        outputs, states, seq_lens = self.build_decoder(
            input_state, helper, reuse=True)

        return outputs, states, seq_lens

    def build_train_loss(self, data_inputs, train_outputs):
        decoder_labels = data_inputs["decoder_labels"]
        decoder_lens = data_inputs["decoder_lens"]
        global_step = tf.train.get_or_create_global_step()
        seq_masks = tf.sequence_mask(decoder_lens, dtype=tf.float32)
        seq_loss = tf.contrib.seq2seq.sequence_loss(train_outputs.rnn_output,
                                                    decoder_labels, seq_masks)
        tf.add_to_collection("train_summaries",
                             tf.summary.scalar("train/seq_loss", seq_loss))
    
        train_op = tf.train.AdamOptimizer(
            learning_rate=self._hparams.init_learning_rate).minimize(
                seq_loss, global_step=global_step)
    
        return seq_loss, train_op
        
    def build_sample_state_ops(self, state_ctr):
        # state_ctr: [state_len]
        # return: [max_hit, state_len]
        search_hits = self._hparams.search_hits
        search_hits.sort()
        max_search_hits = max(search_hits)
        noise = tf.random_normal(
            [max_search_hits, state_ctr.get_shape()[0].value])
        return state_ctr + noise
    
    def build_ctr_summary(self, state_ctr_smile,
                          smiles, summary_cls="val"):
        # Reture None.
        # [1, val_num]
        ctr_tm_hits = self.build_tanimoto_distance_op(
            state_ctr_smile, smiles)
        ctr_best_hits = tf.reduce_max(ctr_tm_hits, axis=0)
        # produce a zero-vector of shape [val_num]
        best_hit_argmax_op = tf.argmax(ctr_tm_hits, axis=0)
        ctr_hit_smiles = tf.gather(
            state_ctr_smile, best_hit_argmax_op, axis=0)
        tf.add_to_collection("%s_summaries" % summary_cls,
            tf.summary.histogram(
                "%s/ctr_TM_hit" % summary_cls,
                ctr_tm_hits))
        tf.add_to_collection("%s_summaries" % summary_cls,
            tf.summary.scalar("%s/ctr_TM_best" % summary_cls,
                              tf.reduce_max(ctr_tm_hits)))
        ctr_best_hit_csv = tf.reduce_join(
            tf.stack(
                [smiles, ctr_hit_smiles, tf.as_string(ctr_best_hits)], 
                axis=1),
            axis=1, separator=",")
        tf.add_to_collection("%s_summaries" % summary_cls,
            tf.summary.text("%s/ctr_TM_best_hit_csv" % summary_cls,
                            ctr_best_hit_csv))
    
    def build_sample_summary(self, sampled_smiles, smiles, summary_cls="val"):
        sample_tm_hits = self.build_tanimoto_distance_op(
            sampled_smiles, smiles)

        for hit_num in self._hparams.search_hits:
            # [hit_num, val_num]
            search_tm_hits = tf.gather(
                sample_tm_hits, tf.range(hit_num), axis=0)
            tf.add_to_collection("%s_summaries" % summary_cls,
                tf.summary.scalar(
                    "%s/sample_TM_best_hit_max_%d" % (summary_cls, hit_num),
                    tf.reduce_max(search_tm_hits)))
            # [val_num]
            hit_max_op = tf.reduce_max(search_tm_hits, axis=0)
            tf.add_to_collection("%s_summaries" % summary_cls,
                tf.summary.scalar(
                    "%s/sample_TM_best_hit_mean_%d" % (summary_cls, hit_num),
                    tf.reduce_mean(hit_max_op)))
            tf.add_to_collection("%s_summaries" % summary_cls,
                tf.summary.histogram("%s/sample_TM_best_hit_%d" % (
                                         summary_cls, hit_num),
                                     tf.reduce_max(search_tm_hits, axis=0)))
            hit_argmax_op = tf.argmax(search_tm_hits, axis=0)
            # [val_num]
            search_best_hit_smiles = tf.gather(
                sampled_smiles, hit_argmax_op, axis=0)
            search_best_hit_csv = tf.reduce_join(
                tf.stack(
                    [smiles, search_best_hit_smiles, tf.as_string(hit_max_op)], 
                    axis=1),
                axis=1, separator=",")
            tf.add_to_collection("%s_summaries" % summary_cls,
                tf.summary.text("%s/sample_TM_best_hit_csv_%d" % (
                                    summary_cls, hit_num),
                                search_best_hit_csv))
        
    
    def build_val_net(self, data_inputs):
        # We calculate the center of validation data.
        smiles = data_inputs["smile"]
        encoder_input = data_inputs["encoder_inputs"]
        _, state = self.build_seq2state_op(encoder_input)
        
        state_ctr = tf.reduce_mean(state, 0)
        seq_outputs, _, _ = self.build_state2seq_op(
            tf.expand_dims(state_ctr, 0))
        state_ctr_seq = seq_outputs.rnn_output
        state_ctr_smile = self.build_seq2smile_op(state_ctr_seq)
        tf.add_to_collection("val_fp_center", state_ctr)
        tf.add_to_collection("val_summaries", 
            tf.summary.text("val/val_center_smile", state_ctr_smile))

        # Add summaries for val central smile.
        self.build_ctr_summary(state_ctr_smile, smiles)


        # Random sample smiles.
        sampled_states = self.build_sample_state_ops(state_ctr)
        sampled_outputs, _, _ = self.build_state2seq_op(sampled_states)
        sampled_smiles = self.build_seq2smile_op(sampled_outputs.rnn_output)

        # validation TM hits. [max_search_hits, val_data_num]
        self.build_sample_summary(sampled_smiles, smiles)


        return state_ctr, state_ctr_smile, sampled_smiles
    
    def build_test_net(self, val_ctr_smile, val_sampled_smiles, data_inputs):
        smiles = data_inputs["smile"]
        
        # Add summaries for val central smile.
        self.build_ctr_summary(
            val_ctr_smile, smiles, summary_cls="test")
        # validation TM hits. [max_search_hits, val_data_num]
        self.build_sample_summary(
            val_sampled_smiles, smiles, summary_cls="test")

    def build_seq2smile_op(self, input_seqs):
        # input_seqs: [batch_size, max_batch_seq_len, vocab_size]
        # [batch_size, max_seq_len]
        seqs = tf.argmax(input_seqs, axis=-1) 
        # From numpy array to strings.
        def recover_seq_repr(token_seqs):
            def recover_single_seq_repr(token_ids):
                res = []
                for token_id in token_ids:
                    if token_id == self._vocab.EOS_ID:
                        break
                    res.append(self._vocab.query_token(token_id))
                return "".join(res)
            return np.apply_along_axis(recover_single_seq_repr, 1, token_seqs)
        return tf.py_func(recover_seq_repr, [seqs], tf.string)

    @staticmethod
    def is_valid_single_smile(smile):
        try:
            mol = MolFromSmiles(smile)
            return mol is not None
        except Exception as ex:
            return False

    def build_is_valid_smile_op(self, input_smiles):
        # input_smile: [batch_size] of tf.string.
        def is_valid_smiles(smiles):
            return np.array(
                [self.is_valid_single_smile(x) for x in smiles], dtype=np.bool)
        valid_smiles = tf.py_func(is_valid_smiles, [input_smiles], tf.bool)
        return valid_smiles
        
    def build_tanimoto_distance_op(
        self, first_smiles, second_smiles, mode="product"):
        # first_smiles: [batch_size] of tf.string.
        # second_smiles: [batch_size] of tf.string.
        # mode: product or zip. 
        #   zip: compare one to one and produce output [batch_size].
        #   product: compare in cross product fashion, output 
        #            [first_size, second_size]

        def compute_tanimoto_metrics(first_smiles, second_smiles, mode):

            def compute_single_tanimoto_metric(first_smile, second_smile):
                if first_smile is None or second_smile is None:
                    return 0.
                first_mol = MolFromSmiles(first_smile)
                second_mol = MolFromSmiles(second_smile)
                if first_mol is None or second_mol is None:
                    return 0.
                tanimoto_similarity = DataStructs.FingerprintSimilarity(
                    FingerprintMols.FingerprintMol(first_mol),
                    FingerprintMols.FingerprintMol(second_mol),
                    metric=DataStructs.TanimotoSimilarity)
                return tanimoto_similarity

            if mode == "zip":
                assert len(first_smiles) == len(second_smiles), (
                    "Zip mode should have same number of smiles to compare.")
            aggr_fn = zip if mode == "zip" else product
            # Pre-filter the valid smile for evals.
            filter_fn = lambda smiles: [
                s if self.is_valid_single_smile(s) else None for s in smiles]
            first_smiles, second_smiles = (
                filter_fn(first_smiles), filter_fn(second_smiles))
            tanimoto_similarities = np.array(
                [
                    compute_single_tanimoto_metric(a, b)
                    for a, b in aggr_fn(first_smiles, second_smiles)
                ],
                dtype=np.float32)
            if mode == "product":
                tanimoto_similarities = tanimoto_similarities.reshape([
                    len(first_smiles), len(second_smiles)])
            return tanimoto_similarities

        tanimoto_similarities = tf.py_func(compute_tanimoto_metrics,
                                           [first_smiles, second_smiles, mode],
                                           tf.float32)
        return tanimoto_similarities
