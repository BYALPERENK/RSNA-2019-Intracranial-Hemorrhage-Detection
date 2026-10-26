import pytest
import torch

from ich.models.sequence_model import SequenceModel

ARCHS = ["gru", "lstm", "tcn", "transformer"]
HIDDEN = {"gru": 256, "lstm": 224, "tcn": 256, "transformer": 256}


def model_of(arch, residual=True):
    torch.manual_seed(0)
    return SequenceModel(0, use_features=False, use_logits=True, hidden=HIDDEN[arch], residual=residual,
                         arch=arch).eval()


def inputs(lengths, t):
    torch.manual_seed(1)
    logits = torch.randn(len(lengths), t, 6)
    for i, n in enumerate(lengths):
        logits[i, n:] = 0.0
    return torch.zeros(len(lengths), t, 0), logits, torch.zeros(len(lengths), t, 0), torch.tensor(lengths)


@pytest.mark.parametrize("arch", ARCHS)
def test_untrained_model_returns_the_cnn_logits(arch):
    features, logits, side, lengths = inputs([20, 60], 60)
    with torch.no_grad():
        out = model_of(arch)(features, logits, side, lengths)
    assert torch.equal(out, logits)


@pytest.mark.parametrize("arch", ARCHS)
def test_padding_does_not_reach_the_real_slices(arch):
    # a nonzero head: with the zero head of the residual model every output equals the input, hiding any leak
    model = model_of(arch, residual=False)
    features, logits, side, lengths = inputs([20, 60], 60)
    logits[0, 20:] = 5.0  # padding with values: must not matter
    with torch.no_grad():
        padded = model(features, logits, side, lengths)[0, :20]
        alone = model(features[:1, :20], logits[:1, :20], side[:1, :20], lengths[:1])[0]
    torch.testing.assert_close(padded, alone, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("arch", ["tcn", "transformer"])
def test_every_slice_sees_the_whole_series(arch):
    # changing the first slice of a 60-slice series changes the output of the last one (the recurrent encoders reach
    # every slice by construction; untrained, the change fades out over the 60 steps)
    model = model_of(arch, residual=False)
    features, logits, side, lengths = inputs([60], 60)
    changed = logits.clone()
    changed[0, 0] += 3.0
    with torch.no_grad():
        a, b = model(features, logits, side, lengths), model(features, changed, side, lengths)
    assert (a[0, -1] - b[0, -1]).abs().max() > 1e-6


def test_parameter_counts_are_matched():
    counts = {arch: sum(p.numel() for p in model_of(arch).parameters()) for arch in ARCHS}
    assert all(abs(n / counts["gru"] - 1) < 0.05 for n in counts.values()), counts


@pytest.mark.parametrize("arch", ["gru", "tcn"])
def test_two_branches_untrained_and_padding(arch):
    torch.manual_seed(0)
    model = SequenceModel(16, use_features=True, use_logits=True, proj=8, hidden=32, dropout=0.5, arch=arch,
                          two_branch=True).eval()
    _, logits, side, lengths = inputs([20, 60], 60)
    features = torch.randn(2, 60, 16)
    with torch.no_grad():
        out = model(features, logits, side, lengths)
    assert torch.equal(out, logits) and torch.equal(model.aux, logits)
    for head in (model.head, model.head1):  # nonzero heads, so that a leak would show
        torch.nn.init.normal_(head.weight), torch.nn.init.normal_(head.bias)
    features[0, 20:], logits[0, 20:] = 5.0, 5.0
    with torch.no_grad():
        padded = model(features, logits, side, lengths)[0, :20]
        aux = model.aux[0, :20]
        alone = model(features[:1, :20], logits[:1, :20], side[:1, :20], lengths[:1])[0]
    torch.testing.assert_close(padded, alone, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(aux, model.aux[0], rtol=1e-4, atol=1e-5)
