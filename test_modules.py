import unittest
import torch
from pkm import ProductKeyMemory
from model import Config, LanguageModel, parameter_counts


class PKMTests(unittest.TestCase):
    def test_shapes_and_gradients(self):
        torch.manual_seed(0)
        layer = ProductKeyMemory(d_model=16, n_values=64, heads=2, top_k=4, d_query=8)
        x = torch.randn(3, 5, 16, requires_grad=True)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        for name, parameter in layer.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        selected_grad = layer.values.grad.abs().sum(dim=1) > 0
        self.assertGreater(selected_grad.sum().item(), 0)
        self.assertLessEqual(selected_grad.sum().item(), 3 * 5 * 2 * 4)

    def test_causal_model_and_ablation_switch(self):
        torch.manual_seed(1)
        cfg = Config(vocab_size=101, d_model=16, layers=2, heads=2, max_seq_len=8, use_pkm=True, pkm_layer=1, n_values=64, pkm_heads=2, top_k=4, d_query=8)
        model = LanguageModel(cfg)
        tokens = torch.randint(0, 101, (2, 8))
        logits, loss = model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 8, 101))
        self.assertTrue(torch.isfinite(loss))
        model.disable_pkm = True
        disabled, _ = model(tokens)
        self.assertFalse(torch.equal(logits, disabled))
        counts = parameter_counts(model)
        self.assertEqual(counts["total"] - counts["active"], 64 * 16)

    def test_recurrence_reuses_weights_and_queries_once(self):
        cfg = Config(vocab_size=101, d_model=16, layers=4, heads=2, max_seq_len=8,
                     use_pkm=True, pkm_layer=2, n_values=64, pkm_heads=2,
                     top_k=4, d_query=8, recurrence=3)
        model = LanguageModel(cfg)
        tokens = torch.randint(0, 101, (2, 8))
        calls = {"pkm": 0, "blocks": [0] * 4}
        model.pkm.register_forward_hook(lambda *args: calls.__setitem__("pkm", calls["pkm"] + 1))
        for i, block in enumerate(model.blocks):
            block.register_forward_hook(lambda *args, i=i: calls["blocks"].__setitem__(i, calls["blocks"][i] + 1))
        logits, _ = model(tokens)
        self.assertEqual(logits.shape, (2, 8, 101))
        self.assertEqual(calls["pkm"], 1)
        self.assertEqual(calls["blocks"], [1, 1, 3, 3])
        cfg.recurrence = 1
        self.assertEqual(parameter_counts(model), parameter_counts(LanguageModel(cfg)))


if __name__ == "__main__":
    unittest.main()
