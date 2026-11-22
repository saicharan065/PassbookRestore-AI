import numpy as np

from passbookrestore import metrics


def test_levenshtein_basic():
    assert metrics.levenshtein("kitten", "sitting") == 3
    assert metrics.levenshtein("", "abc") == 3
    assert metrics.levenshtein("same", "same") == 0


def test_cer_normalises_whitespace():
    assert metrics.cer("Date  Balance\n100.00", "Date Balance 100.00") == 0.0
    assert metrics.cer("abcd", "abxd") == 0.25


def test_word_accuracy_is_order_independent():
    assert metrics.word_accuracy("a b c", "c a b") == 1.0
    assert metrics.word_accuracy("a b c d", "a b") == 0.5
    assert metrics.word_accuracy("", "anything") == 1.0


def test_quality_metrics_respond_to_degradation():
    rng = np.random.default_rng(0)
    clean = np.full((200, 300), 240, np.uint8)
    clean[40:100, 20:280] = 20          # ~26% ink coverage, like a dense text page
    noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)
    assert metrics.noise_level(noisy) > metrics.noise_level(clean)
    faded = (clean * 0.4 + 140).astype(np.uint8)
    assert metrics.contrast(faded) < metrics.contrast(clean)


def test_structure_identical_images_score_one():
    img = np.full((400, 600), 245, np.uint8)
    for y in (50, 150, 250, 350):
        img[y:y + 2, 30:570] = 40
    for x in (30, 300, 570):
        img[50:350, x:x + 2] = 40
    rep = metrics.structure_report(img, img)
    assert rep["structure_score"] > 0.99
    assert rep["h_rules"] == rep["h_rules_ref"] == 4
    assert rep["v_rules"] == rep["v_rules_ref"] == 3
