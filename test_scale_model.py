import unittest
import torch
from scale_model import ScaleConfig, ScaleLanguageModel, SparseAdamW, parameter_counts


class ScaleModelTests(unittest.TestCase):
    def test_forward_backward_and_value_device(self):
        cfg = ScaleConfig(vocab_size=101, d_model=16, layers=2, heads=2, ff_mult=4,
                          max_seq_len=8, n_values=64, pkm_layer=1, pkm_heads=2,
                          top_k=4, d_query=8, pkm_dv=32)
        model = ScaleLanguageModel(cfg)
        tokens = torch.randint(0, 101, (2, 8))
        hidden, loss = model(tokens, tokens)
        self.assertEqual(hidden.shape, (2, 8, 16))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(model.pkm.values.device.type, "cpu")
        self.assertIsNone(model.pkm.values.grad)
        self.assertIsNotNone(model.pkm.last_selected.grad)
        before = model.pkm.values.detach().clone()
        model.collect_value_gradients()
        model.apply_value_gradients(SparseAdamW(model.pkm.values, lr=1e-3))
        self.assertFalse(torch.equal(before, model.pkm.values))
        counts = parameter_counts(model)
        self.assertEqual(counts["value_parameters"], 64 * 32)
        self.assertEqual(counts["value_bytes_fp32"], 64 * 32 * 4)

    def test_dense_and_pkm_disable(self):
        cfg = ScaleConfig(vocab_size=101, d_model=16, layers=2, heads=2, max_seq_len=8,
                          n_values=64, pkm_layer=1, pkm_heads=2, top_k=4, d_query=8, pkm_dv=32)
        model = ScaleLanguageModel(cfg)
        tokens = torch.randint(0, 101, (1, 8))
        first, _ = model(tokens)
        model.disable_pkm = True
        second, _ = model(tokens)
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
